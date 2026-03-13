from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import json
import hashlib
import os
import re
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit
from html.parser import HTMLParser

from fastapi import Body, FastAPI, HTTPException

from llm_gateway import openai_chat_completions, is_sandbox_gateway_base_url

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None


app = FastAPI(title="Autoppia Web Agent API")

_TASK_STATE: dict[str, dict[str, object]] = {}


@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# IWA Selector helpers
# ---------------------------------------------------------------------------

def _sel_attr(attribute: str, value: str, case_sensitive: bool = False) -> Dict[str, Any]:
    return {
        "type": "attributeValueSelector",
        "attribute": attribute,
        "value": value,
        "case_sensitive": case_sensitive,
    }


def _sel_text(value: str, case_sensitive: bool = False) -> Dict[str, Any]:
    return {
        "type": "tagContainsSelector",
        "value": value,
        "case_sensitive": case_sensitive,
    }


def _sel_custom(value: str, case_sensitive: bool = False) -> Dict[str, Any]:
    return {
        "type": "attributeValueSelector",
        "attribute": "custom",
        "value": value,
        "case_sensitive": case_sensitive,
    }


def _sel_xpath(value: str) -> Dict[str, Any]:
    return {
        "type": "xpathSelector",
        "attribute": None,
        "value": value,
        "case_sensitive": False,
    }


def _selector_repr(selector: Dict[str, Any]) -> str:
    t = selector.get("type")
    a = selector.get("attribute")
    v = selector.get("value")
    if t == "attributeValueSelector":
        vv = str(v)
        if len(vv) > 80:
            vv = vv[:77] + "..."
        return f"attr[{a}]={vv}"
    if t == "tagContainsSelector":
        return f"text~={v}"
    return str(selector)


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


class _Candidate:
    def __init__(
        self,
        selector: Dict[str, Any],
        text: str,
        tag: str,
        attrs: Dict[str, str],
        *,
        text_selector: Optional[Dict[str, Any]] = None,
        context: str = "",
        context_raw: str = "",
        group: str = "",
        container_chain: list[str] | None = None,
    ):
        self.selector = selector
        self.text_selector = text_selector
        self.text = text
        self.tag = tag
        self.attrs = attrs
        self.context = context
        self.context_raw = context_raw
        self.group = group
        self.container_chain = container_chain or []

    def click_selector(self) -> Dict[str, Any]:
        if isinstance(self.selector, dict) and self.selector.get("type") == "attributeValueSelector":
            attr = str(self.selector.get("attribute") or "")
            if attr in {"id", "href", "data-testid", "name", "aria-label", "placeholder", "title"}:
                return self.selector

        for a in ("id", "data-testid", "href", "aria-label", "name", "placeholder", "title"):
            v = (self.attrs or {}).get(a)
            if v:
                return _sel_attr(a, v)

        try:
            t = (self.text or "").strip()
            if t and self.tag in {"button", "a"}:
                return _sel_custom(f"{self.tag}:has-text({json.dumps(t)})")
        except Exception:
            pass

        if self.text_selector:
            return self.text_selector

        return self.selector

    def type_selector(self) -> Dict[str, Any]:
        if isinstance(self.selector, dict) and self.selector.get("type") == "attributeValueSelector":
            attr = str(self.selector.get("attribute") or "")
            if attr and attr != "class":
                return self.selector

        for a in ("id", "data-testid", "name", "aria-label", "placeholder", "title"):
            v = (self.attrs or {}).get(a)
            if v:
                return _sel_attr(a, v)

        return _sel_custom(self.tag)


class _CandidateExtractor(HTMLParser):
    """Fallback extractor when BeautifulSoup isn't available."""

    def __init__(self) -> None:
        super().__init__()
        self._current_text: List[str] = []
        self._last_tag: Optional[str] = None
        self._last_attrs: Dict[str, str] = {}
        self.candidates: List[_Candidate] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_map = {k: (v or "") for k, v in attrs}
        self._last_tag = tag
        self._last_attrs = attr_map

        if tag in {"button", "a", "input", "textarea", "select"} or attr_map.get("role") in {"button", "link"}:
            label = attr_map.get("aria-label") or attr_map.get("placeholder") or attr_map.get("title") or ""
            selector = _build_selector(tag, attr_map, text=label)
            group = "FORM" if tag in {"input", "textarea", "select"} else ("LINKS" if tag == "a" else "BUTTONS")
            self.candidates.append(_Candidate(selector, label, tag, attr_map, context="", group=group, container_chain=[group]))

    def handle_data(self, data: str) -> None:
        if self._last_tag in {"button", "a"} and data.strip():
            self._current_text.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag == self._last_tag and self._current_text and self.candidates:
            text = " ".join(self._current_text)[:120]
            c = self.candidates[-1]
            c.text = text or c.text
            if c.tag in {"button", "a"} and c.text:
                c.text_selector = _sel_text(c.text, case_sensitive=False)
        self._current_text = []


def _attrs_to_str_map(attrs: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (attrs or {}).items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            out[k] = " ".join(str(x) for x in v if x is not None).strip()
        else:
            out[k] = str(v)
    return out


def _build_selector(tag: str, attrs: Dict[str, str], *, text: str) -> Dict[str, Any]:
    if attrs.get("id"):
        return _sel_attr("id", attrs["id"])
    if attrs.get("data-testid"):
        return _sel_attr("data-testid", attrs["data-testid"])
    if tag == "a" and attrs.get("href") and not attrs["href"].lower().startswith("javascript:"):
        return _sel_attr("href", attrs["href"])
    if attrs.get("aria-label"):
        return _sel_attr("aria-label", attrs["aria-label"])
    if attrs.get("name"):
        return _sel_attr("name", attrs["name"])
    if attrs.get("placeholder"):
        return _sel_attr("placeholder", attrs["placeholder"])
    if attrs.get("title"):
        return _sel_attr("title", attrs["title"])
    if text and tag in {"button", "a"}:
        return _sel_text(text, case_sensitive=False)
    return _sel_custom(tag)


def _extract_label_from_bs4(soup, el, attr_map: Dict[str, str]) -> str:
    tag = str(getattr(el, "name", "") or "")

    if tag in {"a", "button"}:
        t = _norm_ws(el.get_text(" ", strip=True))
        if t:
            return t[:120]

    for key in ("aria-label", "placeholder", "title"):
        if attr_map.get(key):
            return _norm_ws(attr_map[key])[:120]

    if attr_map.get("aria-labelledby"):
        lid = attr_map["aria-labelledby"].split()[0]
        if lid:
            lab = soup.find(id=lid)
            if lab is not None:
                t = _norm_ws(lab.get_text(" ", strip=True))
                if t:
                    return t[:120]

    if attr_map.get("id"):
        lab = soup.find("label", attrs={"for": attr_map["id"]})
        if lab is not None:
            t = _norm_ws(lab.get_text(" ", strip=True))
            if t:
                return t[:120]

    parent_label = el.find_parent("label")
    if parent_label is not None:
        t = _norm_ws(parent_label.get_text(" ", strip=True))
        if t:
            return t[:120]

    return ""


def _pick_context_container_bs4(el) -> object | None:
    try:
        candidates = []
        cur = el
        for _depth in range(8):
            if cur is None:
                break
            try:
                cur = cur.parent
            except Exception:
                break
            if cur is None:
                break
            tag = str(getattr(cur, "name", "") or "")
            if tag not in {"li", "tr", "article", "section", "div", "td"}:
                continue

            try:
                txt_raw = cur.get_text("\n", strip=True)
            except Exception:
                txt_raw = ""
            L = len(txt_raw or "")
            if L <= 0:
                continue

            try:
                n_inter = len(cur.find_all(["a", "button", "input", "select", "textarea"]))
            except Exception:
                n_inter = 0

            candidates.append((L, n_inter, cur))

        if not candidates:
            return None

        best = None
        best_key = None
        for L, n_inter, node in candidates:
            if not (50 <= L <= 900):
                continue
            if n_inter <= 0 or n_inter > 12:
                continue
            key = (L, n_inter)
            if best is None or key < (best_key or key):
                best = node
                best_key = key
        if best is not None:
            return best

        candidates.sort(key=lambda t: (t[0], t[1]))
        return candidates[0][2]
    except Exception:
        return None


def _container_chain_from_el(soup, el) -> list[str]:
    chain: list[str] = []
    try:
        ancestors = list(el.parents) if hasattr(el, "parents") else []
        for a in reversed(ancestors):
            try:
                tag = str(getattr(a, "name", "") or "")
                if not tag or tag in {"[document]", "html", "body"}:
                    continue
                if tag not in {"header", "nav", "main", "form", "section", "article", "aside", "footer", "ul", "ol", "table", "div"}:
                    continue

                aid = ""
                try:
                    aid = str(a.get("id") or a.get("name") or "").strip()
                except Exception:
                    aid = ""

                role = ""
                try:
                    role = str(a.get("role") or "").strip()
                except Exception:
                    role = ""

                heading = ""
                try:
                    h = a.find(["h1", "h2", "h3"])
                    if h is not None:
                        heading = _norm_ws(h.get_text(" ", strip=True))
                except Exception:
                    heading = ""

                label_bits = [tag]
                if aid:
                    label_bits.append(f"#{aid}")
                if role and role not in {"presentation"}:
                    label_bits.append(f"role={role}")
                if heading:
                    label_bits.append(heading[:50])

                label = " ".join([b for b in label_bits if b])
                label = _norm_ws(label)
                if label and (not chain or chain[-1] != label):
                    chain.append(label)
                if len(chain) >= 4:
                    break
            except Exception:
                continue
    except Exception:
        return chain

    return chain[-3:]


def _extract_candidates_bs4(html: str, *, max_candidates: int) -> List[_Candidate]:
    soup = BeautifulSoup(html, "lxml")

    selectors = [
        "button",
        "a[href]",
        "input",
        "textarea",
        "select",
        "[role='button']",
        "[role='link']",
    ]

    els = []
    for sel in selectors:
        els.extend(soup.select(sel))

    seen: set[tuple[str, str, str]] = set()
    out: List[_Candidate] = []

    for el in els:
        tag = str(getattr(el, "name", "") or "")
        attr_map = _attrs_to_str_map(getattr(el, "attrs", {}) or {})

        group = "PAGE"
        try:
            if el.find_parent("nav") is not None:
                group = "NAV"
            elif el.find_parent("header") is not None:
                group = "HEADER"
            elif el.find_parent("footer") is not None:
                group = "FOOTER"
            elif el.find_parent("form") is not None:
                form = el.find_parent("form")
                fid = ""
                try:
                    fid = str(form.get("id") or form.get("name") or "").strip()
                except Exception:
                    fid = ""
                group = f"FORM:{fid}" if fid else "FORM"
        except Exception:
            pass

        if tag == "input" and attr_map.get("type", "").lower() == "hidden":
            continue
        if attr_map.get("disabled") is not None or attr_map.get("aria-disabled", "").lower() == "true":
            continue

        label = _extract_label_from_bs4(soup, el, attr_map)

        dom_label = label
        context = ""
        context_raw = ""
        title = ""
        try:
            parent = _pick_context_container_bs4(el) or el.find_parent(["li", "tr", "article", "section", "div"])
            if parent is not None:
                context_raw = parent.get_text("\n", strip=True)
                context = _norm_ws(context_raw)
                h = parent.find(["h1", "h2", "h3", "h4"])
                if h is not None:
                    title = _norm_ws(h.get_text(" ", strip=True))
                if not title:
                    t = parent.find(attrs={"class": re.compile(r"title", re.I)})
                    if t is not None:
                        title = _norm_ws(t.get_text(" ", strip=True))
        except Exception:
            context = ""
            context_raw = ""
            title = ""

        if context and len(context) > 180:
            context = context[:177] + "..."

        primary = _build_selector(tag, attr_map, text=(dom_label or label))

        if tag == "select":
            opts: list[tuple[str, str]] = []
            try:
                tmp: list[tuple[str, str]] = []
                for o in el.find_all("option")[:12]:
                    t = ""
                    v = ""
                    try:
                        t = o.get_text(" ", strip=True)
                        v = str(o.get("value") or "").strip()
                    except Exception:
                        t = ""
                        v = ""
                    if t:
                        tmp.append((t, v))
                opts = tmp
            except Exception:
                opts = []

            if isinstance(primary, dict) and primary.get("type") == "attributeValueSelector" and str(primary.get("attribute") or "") == "custom" and str(primary.get("value") or "") == "select":
                first_opt = ""
                try:
                    if opts:
                        first_opt = str(opts[0][0] or "").strip()
                except Exception:
                    first_opt = ""
                if first_opt:
                    safe = first_opt.replace('"', "'")
                    primary = _sel_custom(f'select:has(option:has-text("{safe}"))')

            if opts:
                show: list[str] = []
                for t, v in opts[:8]:
                    if v and v != t:
                        show.append(f"{t} (value={v})")
                    else:
                        show.append(t)
                opt_preview = ", ".join(show)
                label = (dom_label or "select") + f" options=[{opt_preview}]"
                label = label[:200]

        container_chain = []
        try:
            container_chain = _container_chain_from_el(soup, el)
        except Exception:
            container_chain = []

        text_sel = None
        if tag in {"a", "button"} and dom_label:
            text_sel = _sel_text(dom_label, case_sensitive=False)

        sig = (
            str(primary.get("type") or ""),
            str(primary.get("attribute") or ""),
            str(primary.get("value") or ""),
        )
        if sig in seen:
            continue
        seen.add(sig)

        out.append(_Candidate(primary, label, tag, attr_map, text_selector=text_sel, context=context, context_raw=context_raw, group=group, container_chain=container_chain))
        if len(out) >= max_candidates:
            break

    return out


def _extract_candidates(html: str, max_candidates: int = 30) -> List[_Candidate]:
    if not html:
        return []

    if BeautifulSoup is not None:
        try:
            return _extract_candidates_bs4(html, max_candidates=max_candidates)
        except Exception:
            pass

    parser = _CandidateExtractor()
    try:
        parser.feed(html)
    except Exception:
        return []
    return parser.candidates[:max_candidates]


def _summarize_html(html: str, limit: int = 1200) -> str:
    if not html:
        return ""

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "lxml")
            text = _norm_ws(soup.get_text(" ", strip=True))
            return text[:limit]
        except Exception:
            pass

    try:
        text = re.sub(r"<[^>]+>", " ", html)
        return _norm_ws(text)[:limit]
    except Exception:
        return ""


def _dom_digest(html: str, limit: int = 1400) -> str:
    if not html:
        return ""

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "lxml")
            for t in soup(["script", "style", "noscript"]):
                try:
                    t.decompose()
                except Exception:
                    pass

            parts: list[str] = []

            title = ""
            try:
                if soup.title and soup.title.get_text(strip=True):
                    title = _norm_ws(soup.title.get_text(" ", strip=True))
            except Exception:
                title = ""
            if title:
                parts.append(f"TITLE: {title[:160]}")

            heads: list[str] = []
            for h in soup.find_all(["h1", "h2", "h3"], limit=12):
                t = _norm_ws(h.get_text(" ", strip=True))
                if t:
                    heads.append(t[:140])
            if heads:
                parts.append("HEADINGS: " + " | ".join(heads[:10]))

            forms_bits: list[str] = []
            for form in soup.find_all("form", limit=4):
                els = form.find_all(["input", "textarea", "select"], limit=12)
                items: list[str] = []
                for el in els:
                    try:
                        attrs = _attrs_to_str_map(getattr(el, "attrs", {}) or {})
                        itype = (attrs.get("type") or "").lower()
                        lbl = _extract_label_from_bs4(soup, el, attrs)
                        blob = " ".join([lbl, attrs.get("name", ""), attrs.get("id", ""), attrs.get("placeholder", ""), attrs.get("aria-label", ""), itype]).strip()
                        blob = _norm_ws(blob)
                        if not blob:
                            continue
                        items.append(blob[:140])
                    except Exception:
                        continue
                if items:
                    forms_bits.append("; ".join(items[:8]))
            if forms_bits:
                parts.append("FORMS: " + " || ".join(forms_bits[:3]))

            ctas: list[str] = []
            for el in soup.select("button,a[href],[role='button'],[role='link']"):
                try:
                    if len(ctas) >= 14:
                        break
                    t = _norm_ws(el.get_text(" ", strip=True))
                    if not t:
                        t = _norm_ws(str(el.get("aria-label") or "") or "")
                    if not t:
                        continue
                    t_l = t.lower()
                    if t_l in {"home", "logo"}:
                        continue
                    if t not in ctas:
                        ctas.append(t[:90])
                except Exception:
                    continue
            if ctas:
                parts.append("CTAS: " + " | ".join(ctas[:12]))

            out = "\n".join(parts).strip()
            return out[:limit]
        except Exception:
            pass

    return _summarize_html(html, limit=limit)


# ---------------------------------------------------------------------------
# Ranking and prompting
# ---------------------------------------------------------------------------

def _structured_hints(task: str, candidates: List[_Candidate]) -> Dict[str, Any]:
    inputs: List[Dict[str, Any]] = []
    for i, c in enumerate(candidates):
        if c.tag not in {"input", "textarea", "select"}:
            continue
        attrs = {k: (c.attrs.get(k) or "") for k in ("type", "name", "id", "placeholder", "aria-label")}
        label = (c.text or "").strip()
        blob = " ".join([label, c.context or "", attrs.get("name", ""), attrs.get("id", ""), attrs.get("placeholder", ""), attrs.get("aria-label", "")]).lower()

        kind = "text"
        if "password" in blob or attrs.get("type", "").lower() == "password":
            kind = "password"
        elif "email" in blob:
            kind = "email"
        elif any(k in blob for k in ["search", "buscar", "query", "find"]):
            kind = "search"
        elif any(k in blob for k in ["user", "username", "login"]):
            kind = "username"

        inputs.append({
            "candidate_id": i,
            "kind": kind,
            "label": label[:80],
            "required": bool((c.attrs or {}).get("required") is not None),
            "value_len": len(str((c.attrs or {}).get("value") or "")),
            "attrs": {k: v for k, v in attrs.items() if v},
        })
    return {
        "inputs": inputs[:20],
        "clickables": [
            {
                "candidate_id": i,
                "tag": c.tag,
                "label": (c.text or "")[:90],
                "href": (c.attrs or {}).get("href", "") or (c.attrs or {}).get("data-href", ""),
                "context": (c.context or "")[:220],
                "attrs": {k: str((c.attrs or {}).get(k) or "") for k in ("id", "name", "type", "placeholder", "aria-label", "role") if (c.attrs or {}).get(k)},
            }
            for i, c in sorted(
                [(i, c) for i, c in enumerate(candidates) if c.tag in {"a", "button"}],
                key=lambda t: len((t[1].context or "").strip()),
                reverse=True,
            )
        ][:25],
    }


def _score_candidate(task: str, c: _Candidate) -> float:
    score = 0.0

    if c.tag in {"input", "textarea", "select"}:
        score += 6.0
    elif c.tag == "button":
        score += 4.0
    elif c.tag == "a":
        score += 2.0

    attrs = c.attrs or {}
    if attrs.get("id"):
        score += 4.0
    if attrs.get("name"):
        score += 2.0
    if attrs.get("aria-label"):
        score += 2.0
    if attrs.get("placeholder"):
        score += 1.0
    if attrs.get("href"):
        score += 1.0
    if attrs.get("role") in {"button", "link"}:
        score += 0.5

    if attrs.get("required") is not None and c.tag in {"input", "textarea", "select"}:
        score += 2.0

    if c.selector.get("attribute") == "custom" and c.selector.get("value") in {"a", "button", "input", "select", "textarea"}:
        score -= 2.0

    if (c.text or "").strip():
        score += 1.0
    if (c.context or "").strip():
        score += 0.5

    return score


def _select_candidates_for_llm(task: str, candidates_all: List[_Candidate], current_url: str, max_total: int = 60) -> List[_Candidate]:
    if not candidates_all:
        return []

    controls = []
    primaries = []
    contextual = []
    others = []
    for c in candidates_all:
        try:
            from urllib.parse import urlparse
            if c.tag == "a":
                href = str((c.attrs or {}).get("href") or "")
                if href:
                    ph = urlparse(href)
                    pc = urlparse(current_url or "")
                    if ph.path and pc.path and ph.path == pc.path:
                        continue
        except Exception:
            pass
        if c.tag in {"input", "textarea", "select"}:
            controls.append(c)
            continue
        if c.tag == "button":
            primaries.append(c)
            continue
        if c.tag in {"a", "button"} and (c.context or "").strip():
            if len((c.context or "").strip()) >= 40:
                contextual.append(c)
            else:
                others.append(c)
            continue
        others.append(c)

    picked = []
    seen = set()

    def add_many(arr, limit):
        for c in arr:
            sig = f"{_selector_repr(c.selector)}|{(c.text or '')[:80]}"
            if sig in seen:
                continue
            seen.add(sig)
            picked.append(c)
            if len(picked) >= max_total or len(picked) >= limit:
                return

    add_many(controls, max_total)
    if len(picked) < max_total:
        add_many(contextual, max_total)
    if len(picked) < max_total:
        add_many(primaries, max_total)
    if len(picked) < max_total:
        add_many(others, max_total)

    return picked[:max_total]


def _parse_llm_json(content: str) -> Dict[str, Any]:
    if not isinstance(content, str):
        raise ValueError(f"LLM returned non-text content type={type(content)}")

    raw = content.strip()
    try:
        obj = json.loads(raw)
    except Exception:
        s = raw
        if s.startswith("```"):
            s2 = s
            if s2.startswith("```json"):
                s2 = s2[len("```json"):]
            elif s2.startswith("```"):
                s2 = s2[len("```"):]
            if s2.endswith("```"):
                s2 = s2[:-len("```")]
            s = s2.strip()
        start = s.find("{")
        end = s.rfind("}")
        if 0 <= start < end:
            try:
                obj = json.loads(s[start:end + 1])
            except Exception as e:
                raise ValueError(f"LLM returned non-JSON: {raw[:200]}") from e
        else:
            raise ValueError(f"LLM returned non-JSON: {raw[:200]}")
    if not isinstance(obj, dict):
        raise ValueError("LLM returned non-object JSON")
    return obj


def _history_hint(history: List[Dict[str, Any]] | None) -> str:
    if not history:
        return ""

    last = history[-6:]
    repeats = 0
    prev = None
    for h in last:
        k = (str(h.get("action") or ""), h.get("candidate_id"))
        if prev is not None and k == prev and k != ("", None):
            repeats += 1
        prev = k

    if repeats >= 2:
        return "You appear to be repeating the same action. Choose a DIFFERENT candidate or try scroll."

    return ""


def _format_browser_state(*, candidates: List[_Candidate], prev_sig_set: set[str] | None) -> str:
    class _TNode:
        __slots__ = ("name", "children", "items")

        def __init__(self, name: str) -> None:
            self.name = name
            self.children: dict[str, _TNode] = {}
            self.items: list[tuple[int, _Candidate]] = []

    root = _TNode("ROOT")

    def _chain_for(c: _Candidate) -> list[str]:
        ch = []
        try:
            ch = list(getattr(c, "container_chain", []) or [])
        except Exception:
            ch = []
        if not ch:
            g = (getattr(c, "group", "") or "PAGE").strip() or "PAGE"
            ch = [g]
        return [str(x)[:80] for x in ch if str(x).strip()][:3]

    for i, c in enumerate(candidates):
        node = root
        for part in _chain_for(c):
            if part not in node.children:
                node.children[part] = _TNode(part)
            node = node.children[part]
        node.items.append((i, c))

    def _render(node: _TNode, indent: str = "") -> list[str]:
        lines: list[str] = []
        for i, c in node.items:
            label = (c.text or "").strip() or (c.attrs or {}).get("placeholder", "") or (c.attrs or {}).get("aria-label", "")
            label = str(label).strip()

            sig = f"{_selector_repr(c.selector)}|{(c.text or '')[:80]}"
            is_new = bool(prev_sig_set) and (sig not in (prev_sig_set or set()))
            star = "* " if is_new else ""

            attrs_bits: list[str] = []
            for k in ("id", "name", "type", "placeholder", "aria-label", "href", "role"):
                v = (c.attrs or {}).get(k)
                if v:
                    vv = str(v)
                    if len(vv) > 60:
                        vv = vv[:57] + "..."
                    attrs_bits.append(f"{k}={vv}")
            attrs_str = (" | " + ", ".join(attrs_bits)) if attrs_bits else ""

            ctx = ""
            try:
                if c.tag in {"a", "button"} and (c.context or "").strip():
                    ctx = " :: " + _norm_ws(c.context)[:120]
            except Exception:
                ctx = ""

            lines.append(f"{indent}{star}[{i}]<{c.tag}>{label}</{c.tag}>{attrs_str}{ctx}")

        for name, child in node.children.items():
            lines.append(f"{indent}{name}:")
            lines.extend(_render(child, indent + "\t"))

        return lines

    rendered = _render(root, "")
    return "\n".join(rendered)


def _resolve_url(url: str, base_url: str) -> str:
    try:
        from urllib.parse import urljoin
        u = str(url or "").strip()
        b = str(base_url or "").strip()
        if not u:
            return ""
        return urljoin(b, u) if b else u
    except Exception:
        return str(url or "").strip()


def _path_query(url: str, base_url: str = "") -> tuple[str, str]:
    try:
        from urllib.parse import urlparse
        resolved = _resolve_url(url, base_url)
        pu = urlparse(resolved or "")
        return (pu.path or ""), (pu.query or "")
    except Exception:
        s = (url or "").strip()
        return s, ""


def _same_path_query(a: str, b: str, *, base_a: str = "", base_b: str = "") -> bool:
    try:
        return _path_query(a, base_a) == _path_query(b, base_b)
    except Exception:
        return (a or "").strip() == (b or "").strip()


def _preserve_seed_url(target_url: str, current_url: str) -> str:
    try:
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        cur = urlparse(current_url or "")
        tgt = urlparse(target_url or "")
        cur_seed = (parse_qs(cur.query).get("seed") or [None])[0]
        if not cur_seed:
            return target_url
        q = parse_qs(tgt.query)
        if (q.get("seed") or [None])[0] == str(cur_seed):
            return target_url
        q["seed"] = [str(cur_seed)]
        new_q = urlencode(q, doseq=True)
        fixed = tgt._replace(query=new_q)
        if not fixed.scheme and not fixed.netloc:
            return urlunparse(("", "", fixed.path, fixed.params, fixed.query, fixed.fragment))
        return urlunparse(fixed)
    except Exception:
        return target_url


# ---------------------------------------------------------------------------
# HTML inspection tools
# ---------------------------------------------------------------------------

def _safe_truncate(s: str, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else (s[: max(0, n - 3)] + "...")


def _tool_search_text(*, html: str, query: str, regex: bool = False, case_sensitive: bool = False, max_matches: int = 20, context_chars: int = 80) -> Dict[str, Any]:
    q = str(query or "")
    if not q:
        return {"ok": False, "error": "missing query"}

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        if regex:
            pat = re.compile(q, flags)
        else:
            pat = re.compile(re.escape(q), flags)
    except Exception as e:
        return {"ok": False, "error": f"invalid pattern: {str(e)[:120]}"}

    hay = str(html or "")
    out = []
    for m in pat.finditer(hay):
        if len(out) >= int(max_matches or 0):
            break
        a = max(0, m.start() - int(context_chars))
        b = min(len(hay), m.end() + int(context_chars))
        out.append({
            "start": int(m.start()),
            "end": int(m.end()),
            "snippet": _safe_truncate(hay[a:b].replace("\n", " ").replace("\r", " "), 2 * int(context_chars) + 40),
        })

    return {"ok": True, "matches": out, "count": len(out)}


def _tool_css_select(*, html: str, selector: str, max_nodes: int = 25) -> Dict[str, Any]:
    if BeautifulSoup is None:
        return {"ok": False, "error": "bs4 not available"}
    sel = str(selector or "").strip()
    if not sel:
        return {"ok": False, "error": "missing selector"}

    try:
        soup = BeautifulSoup(html or "", "lxml")
        nodes = soup.select(sel)
    except Exception as e:
        return {"ok": False, "error": f"css select failed: {str(e)[:160]}"}

    out = []
    for n in nodes[:int(max_nodes or 0)]:
        try:
            tag = str(getattr(n, "name", "") or "")
            attrs = _attrs_to_str_map(getattr(n, "attrs", {}) or {})
            text = _norm_ws(n.get_text(" ", strip=True))
            out.append({
                "tag": tag,
                "attrs": {k: _safe_truncate(v, 120) for k, v in list(attrs.items())[:12]},
                "text": _safe_truncate(text, 240),
            })
        except Exception:
            continue

    return {"ok": True, "count": len(nodes), "nodes": out}


def _tool_extract_forms(*, html: str, max_forms: int = 10, max_inputs: int = 25) -> Dict[str, Any]:
    if BeautifulSoup is None:
        return {"ok": False, "error": "bs4 not available"}

    try:
        soup = BeautifulSoup(html or "", "lxml")
    except Exception as e:
        return {"ok": False, "error": f"parse failed: {str(e)[:160]}"}

    forms = []
    for f in soup.find_all("form")[:int(max_forms or 0)]:
        try:
            f_attrs = _attrs_to_str_map(getattr(f, "attrs", {}) or {})
            inputs = []
            for el in f.find_all(["input", "textarea", "select", "button"])[:int(max_inputs or 0)]:
                try:
                    tag = str(getattr(el, "name", "") or "")
                    a = _attrs_to_str_map(getattr(el, "attrs", {}) or {})
                    t = _norm_ws(el.get_text(" ", strip=True))
                    inputs.append({
                        "tag": tag,
                        "type": (a.get("type") or "").lower(),
                        "id": a.get("id") or "",
                        "name": a.get("name") or "",
                        "placeholder": a.get("placeholder") or "",
                        "aria_label": a.get("aria-label") or "",
                        "value": _safe_truncate(a.get("value") or "", 120),
                        "text": _safe_truncate(t, 160),
                    })
                except Exception:
                    continue
            forms.append({
                "id": f_attrs.get("id") or "",
                "name": f_attrs.get("name") or "",
                "action": f_attrs.get("action") or "",
                "method": (f_attrs.get("method") or "").upper(),
                "controls": inputs,
            })
        except Exception:
            continue

    return {"ok": True, "forms": forms, "count": len(forms)}


def _tool_xpath_select(*, html: str, xpath: str, max_nodes: int = 25) -> Dict[str, Any]:
    xp = str(xpath or "").strip()
    if not xp:
        return {"ok": False, "error": "missing xpath"}
    try:
        from lxml import html as lxml_html  # type: ignore
    except Exception:
        return {"ok": False, "error": "lxml not available"}

    try:
        doc = lxml_html.fromstring(html or "")
        nodes = doc.xpath(xp)
    except Exception as e:
        return {"ok": False, "error": f"xpath failed: {str(e)[:160]}"}

    out = []
    for n in nodes[:int(max_nodes or 0)]:
        try:
            if not hasattr(n, "tag"):
                out.append({"value": _safe_truncate(str(n), 240)})
                continue
            tag = str(getattr(n, "tag", "") or "")
            attrs = {k: _safe_truncate(str(v), 120) for k, v in list(getattr(n, "attrib", {}) or {}).items()[:12]}
            text = _norm_ws(" ".join(n.itertext()))
            out.append({"tag": tag, "attrs": attrs, "text": _safe_truncate(text, 240)})
        except Exception:
            continue

    return {"ok": True, "count": len(nodes), "nodes": out}


def _tool_visible_text(*, html: str, max_chars: int = 2000) -> Dict[str, Any]:
    if BeautifulSoup is None:
        txt = re.sub(r"<[^>]+>", " ", str(html or ""))
        txt = _norm_ws(txt)
        return {"ok": True, "text": _safe_truncate(txt, int(max_chars or 0))}

    try:
        soup = BeautifulSoup(html or "", "lxml")
        for t in soup(["script", "style", "noscript"]):
            try:
                t.decompose()
            except Exception:
                pass
        txt = _norm_ws(soup.get_text(" ", strip=True))
        return {"ok": True, "text": _safe_truncate(txt, int(max_chars or 0))}
    except Exception as e:
        return {"ok": False, "error": f"extract text failed: {str(e)[:160]}"}


def _tool_list_candidates(*, candidates: List[_Candidate], max_n: int = 80) -> Dict[str, Any]:
    out = []
    for i, c in enumerate((candidates or [])[:int(max_n or 0)]):
        out.append({
            "id": i,
            "tag": c.tag,
            "group": c.group,
            "text": _safe_truncate(c.text or "", 140),
            "context": _safe_truncate(c.context or "", 200),
            "selector": _selector_repr(c.selector) if isinstance(c.selector, dict) else str(c.selector),
            "click": _selector_repr(c.click_selector()),
        })
    return {"ok": True, "count": len(candidates or []), "candidates": out}


def _tool_list_links(
    *,
    html: str,
    base_url: str,
    max_links: int = 60,
    context_max: int = 260,
    href_regex: str = "",
    text_regex: str = "",
) -> Dict[str, Any]:
    if BeautifulSoup is None:
        return {"ok": False, "error": "bs4 not available"}

    try:
        soup = BeautifulSoup(html or "", "lxml")
    except Exception as e:
        return {"ok": False, "error": f"parse failed: {str(e)[:160]}"}

    href_pat = None
    text_pat = None
    try:
        if href_regex:
            href_pat = re.compile(str(href_regex), re.I)
        if text_regex:
            text_pat = re.compile(str(text_regex), re.I)
    except Exception as e:
        return {"ok": False, "error": f"invalid regex: {str(e)[:160]}"}

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for a in soup.select("a[href]"):
        try:
            href = str(a.get("href") or "").strip()
            if not href or href.lower().startswith("javascript:"):
                continue
            if href_pat and not href_pat.search(href):
                continue

            text = _norm_ws(a.get_text(" ", strip=True))
            if not text:
                text = _norm_ws(str(a.get("aria-label") or "") or "")
            if text_pat and not text_pat.search(text):
                continue

            container = _pick_context_container_bs4(a)
            ctx_raw = ""
            if container is not None:
                try:
                    ctx_raw = container.get_text("\n", strip=True)
                except Exception:
                    ctx_raw = ""
            ctx = _safe_truncate(_norm_ws(ctx_raw) if ctx_raw else "", int(context_max or 0))

            resolved = _resolve_url(href, str(base_url or ""))
            resolved = _preserve_seed_url(resolved, str(base_url or ""))

            sig = (resolved or href) + "|" + (text or "")
            if sig in seen:
                continue
            seen.add(sig)

            out.append({
                "href": _safe_truncate(href, 260),
                "url": _safe_truncate(resolved, 320),
                "text": _safe_truncate(text, 160),
                "context": ctx,
            })
            if len(out) >= int(max_links or 0):
                break
        except Exception:
            continue

    return {"ok": True, "count": len(out), "links": out}


def _tool_list_cards(*, candidates: List[_Candidate], max_cards: int = 25, max_text: int = 900, max_actions_per_card: int = 6) -> Dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}

    for i, c in enumerate(candidates or []):
        try:
            if c.tag not in {"a", "button"}:
                sel = c.click_selector()
                if not (isinstance(sel, dict) and sel.get("type") == "attributeValueSelector" and str(sel.get("attribute") or "") == "href"):
                    continue

            key = (c.context_raw or c.context or "").strip()
            if not key:
                key = "(no_context)"

            g = groups.get(key)
            if g is None:
                facts = []
                try:
                    lines = [ln.strip() for ln in str(key or "").splitlines() if ln.strip()]
                    facts = [ln for ln in lines if any(ch.isdigit() for ch in ln)][:6]
                except Exception:
                    facts = []
                g = {"card_text": _safe_truncate(key, int(max_text or 0)), "card_facts": facts, "candidate_ids": [], "actions": []}
                groups[key] = g

            g["candidate_ids"].append(i)
            if len(g["actions"]) < int(max_actions_per_card or 0):
                sel = c.click_selector()
                href = ""
                try:
                    if isinstance(sel, dict) and sel.get("type") == "attributeValueSelector" and str(sel.get("attribute") or "") == "href":
                        href = str(sel.get("value") or "").strip()
                except Exception:
                    href = ""

                g["actions"].append({
                    "candidate_id": i,
                    "tag": c.tag,
                    "text": _safe_truncate(c.text or "", 140),
                    "click": _selector_repr(sel),
                    "href": _safe_truncate(href, 240) if href else "",
                })
        except Exception:
            continue

    ranked = []
    for _k, g in groups.items():
        txt = str(g.get("card_text") or "")
        n_actions = len(g.get("actions") or [])
        L = len(txt)
        penalty = 0
        if L < 40:
            penalty += 400
        if L > 900:
            penalty += min(1200, L - 900)
        score = (1000 - penalty + min(L, 700), n_actions)
        ranked.append((score, g))

    ranked.sort(key=lambda x: x[0], reverse=True)
    cards = [g for _, g in ranked[:int(max_cards or 0)]]
    return {"ok": True, "count": len(cards), "cards": cards}


_TOOL_REGISTRY = {
    "search_text": _tool_search_text,
    "visible_text": _tool_visible_text,
    "css_select": _tool_css_select,
    "xpath_select": _tool_xpath_select,
    "extract_forms": _tool_extract_forms,
    "list_links": _tool_list_links,
    "list_candidates": _tool_list_candidates,
    "list_cards": _tool_list_cards,
}


def _run_tool(tool: str, args: Dict[str, Any], *, html: str, url: str, candidates: List[_Candidate]) -> Dict[str, Any]:
    t = str(tool or "").strip()
    fn = _TOOL_REGISTRY.get(t)
    if fn is None:
        return {"ok": False, "error": f"unknown tool: {t}", "known": sorted(_TOOL_REGISTRY.keys())}

    a = args if isinstance(args, dict) else {}
    if t == "list_candidates":
        return fn(candidates=candidates, **{k: v for k, v in a.items() if k in {"max_n"}})
    if t == "list_cards":
        return fn(candidates=candidates, **{k: v for k, v in a.items() if k in {"max_cards", "max_text", "max_actions_per_card"}})
    if t == "list_links":
        return fn(html=html, base_url=str(url or ""), **{k: v for k, v in a.items() if k in {"max_links", "context_max", "href_regex", "text_regex"}})
    if t in {"search_text", "visible_text", "css_select", "xpath_select", "extract_forms"}:
        return fn(html=html, **a)

    return {"ok": False, "error": f"tool not wired: {t}"}


# ---------------------------------------------------------------------------
# Credential extraction & task classification (IWA-specific helpers)
# ---------------------------------------------------------------------------

def _extract_credentials_from_task(task: str) -> Dict[str, str]:
    """Parse credential values literally embedded in the task prompt.

    IWA replaces <USERNAME>/<PASSWORD> placeholders in the task *before* sending
    to the agent, so the prompt contains real values like:
      'Login where username equals john123 and password equals p@ss!'
    We extract them so the LLM sees them in a structured block.
    """
    creds: Dict[str, str] = {}
    t = task or ""

    patterns = [
        # "username equals X" / "username: X" (up to next comma, 'and', 'or', end)
        (r"(?:user[- _]?name|user)\s*(?:equals|:|is|=)\s*['\"]?([^\s,'\"\n]+)['\"]?", "username"),
        (r"(?:email)\s*(?:equals|:|is|=)\s*['\"]?([^\s,'\"\n]+)['\"]?", "email"),
        (r"(?:password|pass)\s*(?:equals|:|is|=)\s*['\"]?([^\s,'\"\n]+)['\"]?", "password"),
        # signup variants
        (r"signup[_-]?username\s*(?:equals|:|is|=)\s*['\"]?([^\s,'\"\n]+)['\"]?", "signup_username"),
        (r"signup[_-]?email\s*(?:equals|:|is|=)\s*['\"]?([^\s,'\"\n]+)['\"]?", "signup_email"),
        (r"signup[_-]?password\s*(?:equals|:|is|=)\s*['\"]?([^\s,'\"\n]+)['\"]?", "signup_password"),
    ]

    for pat, key in patterns:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            val = m.group(1).strip().rstrip(".,;:")
            if val and val not in creds.values():
                creds[key] = val

    return creds


def _classify_task(task: str) -> str:
    """Return a short label for the task type to prime the LLM."""
    t = (task or "").lower()

    # Multi-step tasks first (order matters)
    if re.search(r"\b(logout|sign.?out|log.?out)\b", t) and re.search(r"\b(login|sign.?in|log.?in)\b", t):
        return "LOGIN_THEN_LOGOUT"
    if re.search(r"\b(add|remove|delete).*(watchlist|reading.?list|wishlist|cart)\b", t) and re.search(r"\b(login|sign.?in)\b", t):
        return "LOGIN_THEN_LIST_ACTION"
    if re.search(r"\b(add|post|submit).*(comment|review|rating)\b", t) and re.search(r"\b(login|sign.?in)\b", t):
        return "LOGIN_THEN_COMMENT"
    if re.search(r"\b(add|insert|create|register).*(film|movie|book)\b", t) and re.search(r"\b(login|sign.?in)\b", t):
        return "LOGIN_THEN_ADD_ITEM"
    if re.search(r"\b(edit|update|modify).*(film|movie|book)\b", t) and re.search(r"\b(login|sign.?in)\b", t):
        return "LOGIN_THEN_EDIT_ITEM"
    if re.search(r"\b(delete|remove).*(film|movie|book)\b", t) and re.search(r"\b(login|sign.?in)\b", t):
        return "LOGIN_THEN_DELETE_ITEM"
    if re.search(r"\b(edit|update|modify).*(profile|account|user)\b", t) and re.search(r"\b(login|sign.?in)\b", t):
        return "LOGIN_THEN_EDIT_PROFILE"
    if re.search(r"\b(purchase|buy|checkout|order)\b", t) and re.search(r"\b(login|sign.?in)\b", t):
        return "LOGIN_THEN_PURCHASE"

    # Single-step tasks
    if re.search(r"\b(register|sign.?up|create.*account|fill.*registration)\b", t):
        return "REGISTRATION"
    if re.search(r"\b(login|sign.?in|log.?in|fill.*login)\b", t):
        return "LOGIN"
    if re.search(r"\b(search|look.?for|find|look.?up)\b", t) and re.search(r"\b(film|movie|book)\b", t):
        return "SEARCH_ITEM"
    if re.search(r"\b(filter|sort)\b", t) and re.search(r"\b(film|movie|book)\b", t):
        return "FILTER_ITEM"
    if re.search(r"\b(navigate|go.?to|view.?detail|detail.?page|film.?page|book.?page|movie.?page)\b", t):
        return "NAVIGATE_DETAIL"
    if re.search(r"\b(share)\b", t) and re.search(r"\b(film|movie|book)\b", t):
        return "SHARE_ITEM"
    if re.search(r"\b(watch.*trailer|play.*trailer|trailer)\b", t):
        return "WATCH_TRAILER"
    if re.search(r"\b(preview|open.*preview)\b", t):
        return "OPEN_PREVIEW"
    if re.search(r"\b(add|put).*(cart|basket)\b", t):
        return "ADD_TO_CART"
    if re.search(r"\b(remove|delete).*(cart|basket)\b", t):
        return "REMOVE_FROM_CART"
    if re.search(r"\b(view|show).*(cart|basket)\b", t):
        return "VIEW_CART"
    if re.search(r"\b(purchase|buy|checkout|order)\b", t):
        return "PURCHASE"
    if re.search(r"\b(contact|send.*message|fill.*contact)\b", t):
        return "CONTACT"
    if re.search(r"\b(add|post|submit).*(comment|review)\b", t):
        return "ADD_COMMENT"
    if re.search(r"\b(watchlist|reading.?list|wishlist)\b", t):
        return "LIST_ACTION"

    return "GENERAL"


_TASK_PLAYBOOKS: Dict[str, str] = {
    "REGISTRATION": (
        "PLAYBOOK: 1) Navigate to register/signup page. "
        "2) Type signup_username (or username) into the username field. "
        "3) Type signup_email (or email) into the email field. "
        "4) Type signup_password (or password) into the password field. "
        "5) Click submit/register button. "
        "Use EXACT credential values from TASK_CREDENTIALS or task text."
    ),
    "LOGIN": (
        "PLAYBOOK: 1) Navigate to login page. "
        "2) Type username into the username/email field EXACTLY as given. "
        "3) Type password into the password field EXACTLY as given. "
        "4) Click login/sign-in submit button."
    ),
    "LOGIN_THEN_LOGOUT": (
        "PLAYBOOK: 1) Navigate to login page. "
        "2) Type username exactly. 3) Type password exactly. 4) Click login submit. "
        "5) After login, find logout/sign-out button (often in nav/profile menu). "
        "6) Click logout."
    ),
    "LOGIN_THEN_LIST_ACTION": (
        "PLAYBOOK: 1) Login (navigate to login, fill credentials, submit). "
        "2) Search or browse to find the specific item matching the criteria. "
        "3) Navigate to that item's detail page. "
        "4) Click the add-to-watchlist/reading-list/cart button, or remove button."
    ),
    "LOGIN_THEN_COMMENT": (
        "PLAYBOOK: 1) Login (navigate to login, fill credentials, submit). "
        "2) Find and navigate to the specific item. "
        "3) Find the comment/review form on the detail page. "
        "4) Type the comment text. 5) Submit."
    ),
    "LOGIN_THEN_ADD_ITEM": (
        "PLAYBOOK: 1) Login (navigate to login, fill credentials, submit). "
        "2) Navigate to admin or add-item page (look for Admin/Add Film/Add Book in nav). "
        "3) Fill ALL fields with EXACT values from task. "
        "4) Submit."
    ),
    "LOGIN_THEN_EDIT_ITEM": (
        "PLAYBOOK: 1) Login. "
        "2) Navigate to item list page (admin or main list). "
        "3) Find the specific item matching the search/filter criteria. "
        "4) Click Edit. 5) Update the specified fields EXACTLY. 6) Submit."
    ),
    "LOGIN_THEN_DELETE_ITEM": (
        "PLAYBOOK: 1) Login. "
        "2) Navigate to item list. "
        "3) Find the specific item. "
        "4) Click Delete. 5) Confirm deletion if prompted."
    ),
    "LOGIN_THEN_EDIT_PROFILE": (
        "PLAYBOOK: 1) Login. "
        "2) Navigate to profile/account/settings page. "
        "3) Update the specified fields EXACTLY. 4) Save."
    ),
    "LOGIN_THEN_PURCHASE": (
        "PLAYBOOK: 1) Login. "
        "2) Find the item and add to cart. "
        "3) Navigate to cart/checkout. "
        "4) Complete checkout form. 5) Submit order."
    ),
    "SEARCH_ITEM": (
        "PLAYBOOK: 1) Find the search bar on the page. "
        "2) Type the search query EXACTLY as given in the task. "
        "3) Submit search (press Enter or click search button). "
        "Do NOT modify the search query."
    ),
    "FILTER_ITEM": (
        "PLAYBOOK: 1) Find filter controls on the page. "
        "2) Select/type the filter criteria EXACTLY as specified. "
        "3) Apply the filter."
    ),
    "NAVIGATE_DETAIL": (
        "PLAYBOOK: 1) Browse or search for items. "
        "2) Use list_cards or list_links tool to find item matching ALL criteria. "
        "3) Click/navigate to that item's detail page. "
        "If you need to filter by criteria, use search or filter controls first."
    ),
    "SHARE_ITEM": (
        "PLAYBOOK: 1) Navigate to the specific item detail page. "
        "2) Find the Share button/icon (often a share icon or 'Share' text). "
        "3) Click it."
    ),
    "WATCH_TRAILER": (
        "PLAYBOOK: 1) Navigate to the specific film/movie detail page. "
        "2) Find the 'Watch Trailer' or 'Trailer' or play button. "
        "3) Click it."
    ),
    "OPEN_PREVIEW": (
        "PLAYBOOK: 1) Navigate to the specific book detail page. "
        "2) Find the 'Open Preview' or 'Preview' button. "
        "3) Click it."
    ),
    "ADD_TO_CART": (
        "PLAYBOOK: 1) Find and navigate to the specific book/item. "
        "2) Click 'Add to Cart' button."
    ),
    "REMOVE_FROM_CART": (
        "PLAYBOOK: 1) Navigate to the cart page. "
        "2) Find the specific item in cart. 3) Click Remove/Delete."
    ),
    "VIEW_CART": (
        "PLAYBOOK: 1) Navigate to the cart page (look for Cart icon in nav)."
    ),
    "PURCHASE": (
        "PLAYBOOK: 1) Add the item to cart. "
        "2) Navigate to cart. 3) Click checkout/purchase button. "
        "4) Fill out purchase form. 5) Submit."
    ),
    "CONTACT": (
        "PLAYBOOK: 1) Navigate to the Contact page (look for Contact in nav). "
        "2) Fill in name, email, message fields with EXACT values from task. "
        "3) Submit the form."
    ),
    "ADD_COMMENT": (
        "PLAYBOOK: 1) Navigate to the specific item detail page. "
        "2) Find the comment/review form. "
        "3) Type the comment EXACTLY as specified. 4) Submit."
    ),
    "LIST_ACTION": (
        "PLAYBOOK: 1) Navigate to the item detail page. "
        "2) Find the watchlist/reading-list button. 3) Click add or remove."
    ),
    "GENERAL": (
        "PLAYBOOK: Analyze the task carefully, identify the key action required, "
        "and execute the most direct path to complete it."
    ),
}


def _llm_decide(
    *,
    task_id: str,
    task: str,
    step_index: int,
    url: str,
    candidates: List[_Candidate],
    page_summary: str,
    dom_digest: str,
    html_snapshot: str,
    history: List[Dict[str, Any]] | None,
    extra_hint: str = "",
    state_delta: str = "",
    prev_sig_set: set[str] | None = None,
    relevant_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    browser_state = _format_browser_state(candidates=candidates, prev_sig_set=prev_sig_set)

    task_type = _classify_task(task)
    playbook = _TASK_PLAYBOOKS.get(task_type, _TASK_PLAYBOOKS["GENERAL"])

    # Extract credentials from both task text and relevant_data
    creds_from_task = _extract_credentials_from_task(task)
    creds_from_data: Dict[str, str] = {}
    if relevant_data and isinstance(relevant_data, dict):
        for k, v in relevant_data.items():
            if isinstance(v, str) and v:
                creds_from_data[str(k)] = str(v)
    all_creds = {**creds_from_task, **creds_from_data}

    system_msg = (
        "You are a web automation agent. You will be given a task, the current browser state, and history. "
        "Return JSON only (no markdown, no explanation). "
        "Return a JSON object with EXACTLY these keys: action, candidate_id, text, url, evaluation_previous_goal, memory, next_goal. "
        "\n"
        "ACTION RULES:\n"
        "- action must be one of: click, type, select, navigate, scroll_down, scroll_up, done\n"
        "- For click/type/select: candidate_id must be an integer (index from BROWSER_STATE list)\n"
        "- For type/select: text must be non-empty\n"
        "- For navigate: url must be a full URL. Preserve existing query params (e.g., ?seed=X)\n"
        "- Use done ONLY when the task is clearly and fully completed\n"
        "\n"
        "STRICT VALUE COPYING (CRITICAL):\n"
        "- Copy ALL values EXACTLY as provided in TASK_CREDENTIALS or the task text\n"
        "- Do NOT correct typos, do NOT remove numbers, do NOT truncate strings\n"
        "- Example: if task says 'Sofia 4', type 'Sofia 4' not 'Sofia'\n"
        "- Example: if task says 'ng', type 'ng' not 'anything'\n"
        "\n"
        "CREDENTIAL HANDLING:\n"
        "- If TASK_CREDENTIALS is shown, use those exact values when typing into username/email/password fields\n"
        "- Map 'username'/'signup_username' → username input field\n"
        "- Map 'email'/'signup_email' → email input field\n"
        "- Map 'password'/'signup_password' → password input field\n"
        "\n"
        "MULTI-STEP TASKS:\n"
        "- For tasks requiring login THEN another action: first complete the full login, then do the secondary action\n"
        "- Track progress in memory: store what you've done and what remains\n"
        "\n"
        "HTML INSPECTION TOOLS (use when needed to find items):\n"
        "Return {\"tool\": \"<name>\", \"args\": {...}} instead of an action. Max 2 tool calls per step.\n"
        "Tools: search_text({query,regex?,case_sensitive?,max_matches?,context_chars?}); "
        "visible_text({max_chars?}); css_select({selector,max_nodes?}); xpath_select({xpath,max_nodes?}); "
        "extract_forms({max_forms?,max_inputs?}); list_links({max_links?,context_max?,href_regex?,text_regex?}); "
        "list_candidates({max_n?}); list_cards({max_cards?,max_text?,max_actions_per_card?})."
    )

    history_lines: List[str] = []
    for h in (history or [])[-6:]:
        step = h.get("step", "?")
        action = h.get("action", "")
        cid = h.get("candidate_id")
        text = h.get("text", "")
        ok = h.get("exec_ok", True)
        err = h.get("error")
        suffix = "OK" if ok else f"FAILED err={str(err)[:80]}"
        history_lines.append(f"{step}. {action} cid={cid} text={text} [{suffix}]")

    hint = _history_hint(history)

    structured = _structured_hints(task, candidates)

    cards_preview = ""
    try:
        cards_obj = _tool_list_cards(candidates=candidates, max_cards=12, max_text=420, max_actions_per_card=3)
        if isinstance(cards_obj, dict) and cards_obj.get("ok") and cards_obj.get("cards"):
            cards_preview = json.dumps(cards_obj.get("cards"), ensure_ascii=True)
            if len(cards_preview) > 2400:
                cards_preview = cards_preview[:2397] + "..."
    except Exception:
        cards_preview = ""

    agent_mem = ""
    try:
        st2 = _TASK_STATE.get(task_id) if task_id else None
        if isinstance(st2, dict):
            pm = str(st2.get("memory") or "").strip()
            pg = str(st2.get("next_goal") or "").strip()
            if pm or pg:
                agent_mem = f"PREVIOUS MEMORY: {pm}\nPREVIOUS NEXT_GOAL: {pg}\n"
    except Exception:
        agent_mem = ""

    creds_block = ""
    if all_creds:
        creds_block = "TASK_CREDENTIALS (use EXACTLY as-is, no modifications):\n"
        for k, v in all_creds.items():
            creds_block += f"  {k}: {v}\n"

    user_msg = (
        f"TASK: {task}\n"
        f"TASK_TYPE: {task_type}\n"
        f"STEP: {int(step_index)}\n"
        f"URL: {url}\n\n"
        + (creds_block + "\n" if creds_block else "")
        + f"{playbook}\n\n"
        + f"CURRENT STATE (TEXT SUMMARY):\n{page_summary}\n\n"
        + (f"DOM DIGEST (STRUCTURED):\n{dom_digest}\n\n" if dom_digest else "")
        + (f"CARDS (GROUPED CLICKABLE CONTEXTS JSON):\n{cards_preview}\n\n" if cards_preview else "")
        + f"STRUCTURED STATE (JSON):\n{json.dumps(structured, ensure_ascii=True)}\n\n"
        + (f"HISTORY (last steps):\n{chr(10).join(history_lines)}\n\n" if history_lines else "")
        + (f"STATE HINT: {extra_hint}\n\n" if extra_hint else "")
        + (f"AGENT MEMORY:\n{agent_mem}\n" if agent_mem else "")
        + (f"STATE DELTA (prev -> current): {state_delta}\n\n" if state_delta else "")
        + "BROWSER_STATE (interactive elements):\n" + browser_state + "\n\n"
        + "Return ONE JSON action for this step only. "
        + "Provide evaluation_previous_goal, memory, next_goal (1 sentence each)."
    )

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "500"))

    usages: List[Dict[str, Any]] = []
    tool_calls = 0
    max_tool_calls = int(os.getenv("AGENT_MAX_TOOL_CALLS", "2"))

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    def _call(extra_system: str = "") -> Dict[str, Any]:
        sys_msg = system_msg + (" " + extra_system if extra_system else "")
        msgs = [{"role": "system", "content": sys_msg}] + [m for m in messages if m.get("role") != "system"]
        resp = openai_chat_completions(
            task_id=task_id,
            messages=msgs,
            model=str(model),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            u = resp.get("usage")
            if isinstance(u, dict):
                usages.append(u)
        except Exception:
            pass
        content = resp["choices"][0]["message"]["content"]
        obj = _parse_llm_json(content)
        try:
            obj["_meta"] = {"llm_calls": len(usages), "llm_usages": usages, "model": str(model), "tool_calls": tool_calls}
        except Exception:
            pass
        return obj

    def _valid_action(obj: Dict[str, Any]) -> bool:
        a = (obj.get("action") or "").lower()
        if a not in {"click", "type", "select", "navigate", "scroll_down", "scroll_up", "done"}:
            return False
        if a == "navigate":
            u = obj.get("url")
            if not isinstance(u, str) or not u.strip():
                return False
            try:
                if _same_path_query(str(u).strip(), str(url).strip(), base_a=str(url).strip(), base_b=""):
                    return False
            except Exception:
                if str(u).strip() == str(url).strip():
                    return False
            return True
        if a in {"click", "type", "select"}:
            cid = obj.get("candidate_id")
            if isinstance(cid, str) and cid.isdigit():
                cid = int(cid)
            if not isinstance(cid, int) or not (0 <= cid < len(candidates)):
                return False
            if a in {"type", "select"}:
                t = obj.get("text")
                if not isinstance(t, str) or not t.strip():
                    return False
        return True

    def _is_tool(obj: Dict[str, Any]) -> bool:
        t = obj.get("tool")
        if not isinstance(t, str) or not t.strip():
            return False
        if obj.get("action"):
            return False
        return True

    last_obj: Dict[str, Any] = {}
    for _ in range(max_tool_calls + 2):
        try:
            obj = _call()
        except Exception:
            obj = _call("Return ONLY valid JSON. No markdown. No commentary.")

        last_obj = obj

        if _is_tool(obj) and tool_calls < max_tool_calls:
            tool = str(obj.get("tool") or "").strip()
            args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
            tool_calls += 1
            try:
                result = _run_tool(tool, args, html=html_snapshot, url=str(url), candidates=candidates)
            except Exception as e:
                result = {"ok": False, "error": str(e)[:200]}

            messages.append({"role": "assistant", "content": json.dumps({"tool": tool, "args": args}, ensure_ascii=True)})
            messages.append({"role": "user", "content": "TOOL_RESULT " + tool + ": " + json.dumps(result, ensure_ascii=True)})
            continue

        if _valid_action(obj):
            try:
                obj["_meta"] = {"llm_calls": len(usages), "llm_usages": usages, "model": str(model), "tool_calls": tool_calls}
            except Exception:
                pass
            return obj

        obj = _call(
            "Your previous JSON was invalid. Fix it. "
            f"candidate_id must be an integer in [0, {len(candidates) - 1}]. "
            "If action is type/select you must include non-empty text. "
            "If stuck, scroll_down."
        )
        if _valid_action(obj):
            try:
                obj["_meta"] = {"llm_calls": len(usages), "llm_usages": usages, "model": str(model), "tool_calls": tool_calls}
            except Exception:
                pass
            return obj

    return last_obj


def _update_task_state(task_id: str, url: str, sig: str) -> None:
    if not task_id:
        return
    try:
        st = _TASK_STATE.get(task_id)
        if not isinstance(st, dict):
            st = {}
            _TASK_STATE[task_id] = st
        last_sig = str(st.get("last_sig") or "")
        last_url = str(st.get("last_url") or "")
        if sig and sig == last_sig and str(url) == last_url:
            st["repeat"] = int(st.get("repeat") or 0) + 1
        else:
            st["repeat"] = 0
        st["last_sig"] = str(sig)
        st["last_url"] = str(url)
    except Exception:
        return


def _compute_state_delta(
    *,
    task_id: str,
    url: str,
    page_summary: str,
    dom_digest: str,
    html_snapshot: str,
    candidates: List[_Candidate],
) -> str:
    if not task_id:
        return ""

    try:
        st = _TASK_STATE.get(task_id)
        if not isinstance(st, dict):
            st = {}
            _TASK_STATE[task_id] = st

        prev_url = str(st.get("prev_url") or "")
        prev_summary = str(st.get("prev_summary") or "")
        prev_digest = str(st.get("prev_digest") or "")
        prev_sig_set = set(st.get("prev_sig_set") or [])

        cur_sig_set = set()
        for c in candidates[:30]:
            sig = f"{_selector_repr(c.selector)}|{(c.text or '')[:80]}"
            cur_sig_set.add(sig)

        added = len(cur_sig_set - prev_sig_set) if prev_sig_set else len(cur_sig_set)
        removed = len(prev_sig_set - cur_sig_set) if prev_sig_set else 0
        unchanged = len(cur_sig_set & prev_sig_set) if prev_sig_set else 0

        ps = _norm_ws(prev_summary)
        cs = _norm_ws(page_summary)
        pd = _norm_ws(prev_digest)
        cd = _norm_ws(dom_digest)

        same_summary = bool(ps and cs and ps[:240] == cs[:240])
        same_digest = bool(pd and cd and pd[:240] == cd[:240])

        st["prev_url"] = str(url)
        st["prev_summary"] = str(page_summary)
        st["prev_digest"] = str(dom_digest)
        st["prev_sig_set"] = list(cur_sig_set)

        parts = [
            f"url_changed={str(prev_url != str(url)).lower()}" if prev_url else "url_changed=unknown",
            f"summary_changed={str(not same_summary).lower()}" if (ps and cs) else "summary_changed=unknown",
            f"digest_changed={str(not same_digest).lower()}" if (pd and cd) else "digest_changed=unknown",
            f"candidate_added={added}",
            f"candidate_removed={removed}",
            f"candidate_unchanged={unchanged}",
        ]
        return ", ".join(parts)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# HTTP entrypoint
# ---------------------------------------------------------------------------

@app.post("/act", summary="Decide next agent actions")
async def act(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    def _resp(actions: list[dict[str, Any]], metrics: dict[str, Any] | None = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"actions": actions}
        if return_metrics and metrics is not None:
            out["metrics"] = metrics
        return out

    task_id = str(payload.get("task_id") or "")
    task = payload.get("prompt") or payload.get("task_prompt") or ""
    url = payload.get("url") or ""
    step_index = int(payload.get("step_index") or 0)
    return_metrics = os.getenv("AGENT_RETURN_METRICS", "0").lower() in {"1", "true", "yes"}
    html = payload.get("snapshot_html") or ""
    history = payload.get("history") if isinstance(payload.get("history"), list) else None
    relevant_data = payload.get("relevant_data") if isinstance(payload.get("relevant_data"), dict) else None
    page_summary = _summarize_html(html)
    dom_digest = _dom_digest(html)
    task = str(task or "")
    task_for_llm = task

    candidates = _extract_candidates(html, max_candidates=80)
    candidates_all = list(candidates)
    candidates = _select_candidates_for_llm(task, candidates_all, current_url=str(url), max_total=60)

    if task_id == "check":
        if candidates:
            return _resp([{"type": "ClickAction", "selector": candidates[0].click_selector()}], {"decision": "check_click", "candidate_id": 0})
        return _resp([{"type": "WaitAction", "time_seconds": 0.1}], {"decision": "check_wait"})

    st = _TASK_STATE.get(task_id) if task_id else None
    effective_url = str(url)
    try:
        if isinstance(st, dict):
            eu = str(st.get("effective_url") or "").strip()
            if eu:
                effective_url = eu
    except Exception:
        effective_url = str(url)

    extra_hint = ""
    prev_sig_set = None
    try:
        if isinstance(st, dict):
            prev = st.get("prev_sig_set")
            if isinstance(prev, list):
                prev_sig_set = set(str(x) for x in prev)
    except Exception:
        prev_sig_set = None

    state_delta = _compute_state_delta(task_id=task_id, url=str(url), page_summary=page_summary, dom_digest=dom_digest, html_snapshot=html, candidates=candidates)
    try:
        if isinstance(st, dict):
            last_url = str(st.get("last_url") or "")
            repeat = int(st.get("repeat") or 0)
            if last_url and last_url == str(url) and repeat >= 2:
                extra_hint = "You appear stuck on the same URL after repeating an action. Choose a different element or scroll."
    except Exception:
        extra_hint = ""

    try:
        base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        if not os.getenv("OPENAI_API_KEY") and not is_sandbox_gateway_base_url(base_url):
            raise RuntimeError("OPENAI_API_KEY not set")
        decision = _llm_decide(
            task_id=task_id,
            task=task_for_llm,
            step_index=step_index,
            url=effective_url,
            candidates=candidates,
            page_summary=page_summary,
            dom_digest=dom_digest,
            html_snapshot=html,
            history=history,
            extra_hint=extra_hint,
            state_delta=state_delta,
            prev_sig_set=prev_sig_set,
            relevant_data=relevant_data,
        )
        if os.getenv("AGENT_LOG_DECISIONS", "0").lower() in {"1", "true", "yes"}:
            try:
                top = []
                for i, c in enumerate(candidates[:5]):
                    top.append({
                        "i": i,
                        "tag": c.tag,
                        "text": (c.text or "")[:80],
                        "context": (c.context or "")[:80],
                        "sel": _selector_repr(c.selector),
                        "click_sel": _selector_repr(c.click_selector()),
                    })
                print(json.dumps({
                    "task_id": task_id,
                    "url": url,
                    "task": task_for_llm[:200],
                    "decision": decision,
                    "top_candidates": top,
                }, ensure_ascii=True))
            except Exception:
                pass
    except Exception as e:
        if os.getenv("AGENT_DEBUG_ERRORS", "0").lower() in {"1", "true", "yes"}:
            raise HTTPException(status_code=500, detail=str(e)[:400])
        if task_id != "check" and os.getenv("AGENT_LOG_ERRORS", "0").lower() in {"1", "true", "yes"}:
            try:
                key = os.getenv("OPENAI_API_KEY", "")
                key_fpr = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12] if key else "missing"
                base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
                print(json.dumps({"event": "agent_error", "task_id": task_id, "error": str(e)[:400], "key_fpr": key_fpr, "base_url": base_url}, ensure_ascii=True))
            except Exception:
                pass
        return _resp([{"type": "WaitAction", "time_seconds": 1.0}], {"decision": "error_wait"})

    try:
        if task_id:
            st3 = _TASK_STATE.get(task_id)
            if isinstance(st3, dict):
                if isinstance(decision.get("memory"), str):
                    st3["memory"] = decision.get("memory")
                if isinstance(decision.get("next_goal"), str):
                    st3["next_goal"] = decision.get("next_goal")
    except Exception:
        pass

    action = (decision.get("action") or "").lower()
    cid = decision.get("candidate_id")
    text = decision.get("text")

    if isinstance(cid, str) and cid.isdigit():
        cid = int(cid)

    if action == "navigate":
        nav_url_raw = str(decision.get("url") or "").strip()
        if not nav_url_raw:
            return _resp([{"type": "WaitAction", "time_seconds": 1.0}], {"decision": "navigate_missing_url"})

        nav_url = _resolve_url(nav_url_raw, effective_url or str(url))

        if _same_path_query(nav_url, effective_url, base_a=effective_url, base_b=""):
            _update_task_state(task_id, str(url), "navigate_same_url_scroll")
            return _resp([{"type": "ScrollAction", "down": True, "up": False}], {"decision": "scroll_override"})

        _update_task_state(task_id, str(url), f"navigate:{nav_url}")
        try:
            if task_id and isinstance(_TASK_STATE.get(task_id), dict):
                _TASK_STATE[task_id]["effective_url"] = str(nav_url)
        except Exception:
            pass
        return _resp(
            [{"type": "NavigateAction", "url": nav_url, "go_back": False, "go_forward": False}],
            {"decision": "navigate", "url": nav_url, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})},
        )

    if action in {"scroll_down", "scroll_up"}:
        _update_task_state(task_id, str(url), f"{action}")
        return _resp(
            [{"type": "ScrollAction", "down": action == "scroll_down", "up": action == "scroll_up"}],
            {"decision": decision.get("action"), "candidate_id": int(cid) if isinstance(cid, int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})},
        )

    if action == "wait":
        if candidates:
            selector = candidates[0].click_selector()
            _update_task_state(task_id, str(url), f"click_override:{_selector_repr(selector)}")
            return _resp([{"type": "ClickAction", "selector": selector}], {"decision": "click", "candidate_id": int(cid) if isinstance(cid, int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
        _update_task_state(task_id, str(url), "scroll_override")
        return _resp([{"type": "ScrollAction", "down": True, "up": False}], {"decision": "scroll_override", "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})

    if action == "done":
        _update_task_state(task_id, str(url), "done")
        return _resp([], {"decision": "done", "candidate_id": int(cid) if isinstance(cid, int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})

    if action in {"click", "type", "select"} and isinstance(cid, int) and 0 <= cid < len(candidates):
        c = candidates[cid]

        if action == "click":
            selector = c.click_selector()
            try:
                if isinstance(selector, dict) and selector.get("type") == "attributeValueSelector" and selector.get("attribute") == "href":
                    href = str(selector.get("value") or "")
                    fixed = _preserve_seed_url(href, effective_url or str(url))
                    if fixed and fixed != href:
                        fixed_abs = _resolve_url(fixed, effective_url or str(url))
                        if _same_path_query(fixed_abs, effective_url, base_a=effective_url, base_b=""):
                            _update_task_state(task_id, str(url), "navigate_seed_fix_same_url_scroll")
                            return _resp([{"type": "ScrollAction", "down": True, "up": False}], {"decision": "scroll_override"})
                        _update_task_state(task_id, str(url), f"navigate_seed_fix:{fixed_abs}")
                        try:
                            if task_id and isinstance(_TASK_STATE.get(task_id), dict):
                                _TASK_STATE[task_id]["effective_url"] = str(fixed_abs)
                        except Exception:
                            pass
                        return _resp(
                            [{"type": "NavigateAction", "url": fixed_abs, "go_back": False, "go_forward": False}],
                            {"decision": "navigate", "url": fixed_abs, "candidate_id": int(cid) if isinstance(cid, int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})},
                        )
            except Exception:
                pass
            _update_task_state(task_id, str(url), f"click:{_selector_repr(selector)}")
            return _resp(
                [{"type": "ClickAction", "selector": selector}],
                {"decision": "click", "candidate_id": int(cid) if isinstance(cid, int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})},
            )

        if action == "type":
            if not text:
                raise HTTPException(status_code=400, detail="type action missing text")
            selector = c.type_selector()
            _update_task_state(task_id, str(url), f"type:{_selector_repr(selector)}")
            return _resp(
                [{"type": "TypeAction", "selector": selector, "text": str(text)}],
                {"decision": "type", "candidate_id": int(cid) if isinstance(cid, int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})},
            )

        if action == "select":
            if not text:
                raise HTTPException(status_code=400, detail="select action missing text")
            selector = c.type_selector()
            _update_task_state(task_id, str(url), f"select:{_selector_repr(selector)}")
            return _resp(
                [{"type": "SelectDropDownOptionAction", "selector": selector, "text": str(text), "timeout_ms": int(os.getenv("AGENT_SELECT_TIMEOUT_MS", "4000"))}],
                {"decision": "select", "candidate_id": int(cid) if isinstance(cid, int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})},
            )

    if candidates and step_index < 5:
        selector = candidates[0].click_selector()
        _update_task_state(task_id, str(url), f"fallback_click:{_selector_repr(selector)}")
        return _resp(
            [{"type": "ClickAction", "selector": selector}],
            {"decision": "click_override", "candidate_id": 0 if candidates else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})},
        )
    _update_task_state(task_id, str(url), "fallback_wait")
    return _resp([{"type": "WaitAction", "time_seconds": 2.0}], {"decision": "fallback_wait", "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})


@app.post("/step", summary="Alias for /act")
async def step(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return await act(payload)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=True)
