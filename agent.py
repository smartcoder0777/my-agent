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
    """Parse credential and required-field values literally embedded in the task prompt.

    IWA replaces <USERNAME>/<PASSWORD> placeholders in the task *before* sending
    to the agent, so the prompt contains real values like:
      'Login where username equals john123 and password equals p@ss!'
    We extract them so the LLM sees them in a structured block.
    Only extracts EXACT (equals) values, not NOT/not-equals constraints.
    """
    creds: Dict[str, str] = {}
    t = task or ""

    # Helper to extract a field that uses the "equals" operator (not "not equals")
    def _grab_equals(field_pat: str, key: str) -> None:
        # Match "field [not] equals value" - skip if preceded by "not"
        m = re.search(
            r"(?<!\bnot\s)(?<!\bNOT\s)" + field_pat + r"\s+(?:equals?|:|is)\s+['\"]?([^\s,'\"\n\]]+)['\"]?",
            t, re.IGNORECASE
        )
        if m:
            val = m.group(1).strip().rstrip(".,;:")
            if val and key not in creds:
                creds[key] = val

    patterns = [
        # Credentials
        (r"(?:user[- _]?name|user)", "username"),
        (r"(?:email)", "email"),
        (r"(?:password|pass)", "password"),
        (r"signup[_-]?username", "signup_username"),
        (r"signup[_-]?email", "signup_email"),
        (r"signup[_-]?password", "signup_password"),
        # Payment fields
        (r"cvv", "cvv"),
        (r"(?:zipcode|zip)", "zipcode"),
        (r"country", "country"),
        # Task/form fields
        (r"priority", "priority"),
        (r"guests?(?:_set)?", "guests"),
        (r"rating", "rating"),
        (r"reviews?", "reviews"),
    ]

    def _extract_field_equals(field_pat: str) -> Optional[str]:
        """Extract exact value for field=X, preserving spaces inside quotes."""
        # Priority 1: single-quoted value (e.g., username equals 'user ')
        for pat in [
            r"(?<!\w)" + field_pat + r"\s+equals?\s+'([^']*)'",
            r"(?<!\w)" + field_pat + r'\s+equals?\s+"([^"]*)"',
            r"(?<!\w)" + field_pat + r"\s+equals?\s+([^\s,'\"\n\]]+)",
        ]:
            mm = re.search(pat, t, re.IGNORECASE)
            if mm:
                prefix = t[max(0, mm.start()-5):mm.start()].lower()
                if "not" in prefix:
                    continue
                return mm.group(1).rstrip(".,;:")
        return None

    for field_pat, key in patterns:
        if key not in creds:
            val = _extract_field_equals(field_pat)
            if val is not None:
                creds[key] = val

    # Special: "writing a title of job for 'X'" OR "title of job for 'X'"
    m = re.search(r"writing\s+a\s+(?:strong\s+)?title\s+of\s+(?:the\s+)?job\s+for\s+'([^']+)'", t, re.IGNORECASE)
    if not m:
        m = re.search(r"title\s+of\s+(?:the\s+)?job\s+for\s+'([^']+)'", t, re.IGNORECASE)
    if m:
        creds["job_title"] = m.group(1)

    # Special: job title from CONTAINS constraint "query CONTAINS 'X'" (write any title containing X)
    m2 = re.search(r"job\s+posting.*?query\s+CONTAINS\s+'([^']+)'", t, re.IGNORECASE)
    if not m2:
        m2 = re.search(r"writing.*?title.*?(?:query\s+)?CONTAINS\s+'([^']+)'", t, re.IGNORECASE)
    if m2 and "job_title" not in creds:
        creds["job_title_contains"] = m2.group(1)

    # Colon-separated credentials: "username:' '" or "username: ' '"
    for field_label, key in [("username", "username"), ("email", "email"), ("password", "password")]:
        if key not in creds:
            m3 = re.search(
                r"\b" + field_label + r"\s*:\s*'([^']*)'",
                t, re.IGNORECASE
            )
            if m3:
                creds[key] = m3.group(1)

    # Handle <username>/<password> literal placeholders (validator forgot to substitute)
    # Fall back to well-known IWA defaults: user / Passw0rd!
    for placeholder, key, default in [
        ("<username>", "username", "user"),
        ("<password>", "password", "Passw0rd!"),
        ("<web_agent_id>", "web_agent_id", "1"),
    ]:
        if placeholder in t:
            if key not in creds or creds.get(key, "").startswith("<"):
                creds[key] = default

    # Handle "user<web_agent_id>" pattern (replace placeholder with '1')
    for key in ("username", "email", "signup_username", "signup_email"):
        if key in creds and "<web_agent_id>" in creds[key]:
            creds[key] = creds[key].replace("<web_agent_id>", "1")

    return creds


def _parse_task_constraints(task: str) -> List[Dict[str, Any]]:
    """Parse ALL field/operator/value constraints from an IWA task prompt.

    Handles operators: equals, not_equals, contains, not_contains,
    greater_than, less_than, not_in, in.
    Returns list of dicts: {field, op, value}
    """
    constraints: List[Dict[str, Any]] = []
    t = task or ""

    # --- "is not one of [...]" ---
    not_in_pat = re.compile(
        r"([\w_]+(?:\s+[\w_]+)?)\s+is\s+not\s+one\s+of\s+\[([^\]]+)\]",
        re.IGNORECASE
    )
    skip_spans: List[tuple] = []
    for m in not_in_pat.finditer(t):
        field = m.group(1).strip().lower().replace(" ", "_")
        vals = [v.strip().strip("'\"") for v in m.group(2).split(",")]
        constraints.append({"field": field, "op": "not_in", "value": vals})
        skip_spans.append((m.start(), m.end()))

    # --- "is one of [...]" ---
    in_pat = re.compile(
        r"([\w_]+(?:\s+[\w_]+)?)\s+is\s+one\s+of\s+\[([^\]]+)\]",
        re.IGNORECASE
    )
    for m in in_pat.finditer(t):
        field = m.group(1).strip().lower().replace(" ", "_")
        vals = [v.strip().strip("'\"") for v in m.group(2).split(",")]
        constraints.append({"field": field, "op": "in", "value": vals})
        skip_spans.append((m.start(), m.end()))

    def _in_skip(start: int, end: int) -> bool:
        return any(start >= s and end <= e for s, e in skip_spans)

    # --- basic operators (ordered: more specific first) ---
    # Patterns capture a single clean word (or word_word) as field name.
    # Also handle IWA's verbose form "subject that CONTAINS 'X'" and
    # "email address that does NOT CONTAIN 'Y'"
    _FLD = r"([\w]+(?:_[\w]+)*)"  # one or more snake_case words (no spaces)

    basic: List[tuple] = [
        # IWA verbose: "field that does NOT CONTAIN 'value'"
        (_FLD + r"(?:\s+that)?\s+does\s+NOT\s+CONTAIN\s+['\"]([^'\"]+)['\"]", "not_contains"),
        (_FLD + r"(?:\s+that)?\s+does\s+NOT\s+CONTAIN\s+([^\s,'\"\n]+)", "not_contains"),
        # "field not contains 'value'"
        (_FLD + r"\s+not\s+contains?\s+['\"]([^'\"]+)['\"]", "not_contains"),
        (_FLD + r"\s+not\s+contains?\s+([^\s,'\"\n]+)", "not_contains"),
        # "field not equals 'value'"
        (_FLD + r"\s+not\s+equals?\s+['\"]([^'\"]+)['\"]", "not_equals"),
        (_FLD + r"\s+not\s+equals?\s+([^\s,'\"\n]+)", "not_equals"),
        # IWA verbose: "field that CONTAINS 'value'"
        (_FLD + r"(?:\s+that)?\s+CONTAINS\s+['\"]([^'\"]+)['\"]", "contains"),
        # "field contains 'value'"
        (_FLD + r"\s+contains?\s+['\"]([^'\"]+)['\"]", "contains"),
        (_FLD + r"\s+contains?\s+([^\s,'\"\n]+)", "contains"),
        # "field equals 'value'"
        (_FLD + r"\s+equals?\s+['\"]([^'\"]+)['\"]", "equals"),
        (_FLD + r"\s+EQUALS\s+['\"]([^'\"]+)['\"]", "equals"),
        (_FLD + r"\s+equals?\s+([^\s,'\"\n\]]+)", "equals"),
        # greater/less than or equal to (must come BEFORE plain greater/less than)
        (_FLD + r"\s+(?:is\s+)?(?:greater\s+(?:than\s+)?or\s+equal\s+to|greater\s+equal(?:\s+to)?|GREATER\s+EQUAL(?:\s+TO)?|>=)\s+['\"]?([^\s,'\"\n\]]+)['\"]?", "greater_equal"),
        (_FLD + r"\s+(?:is\s+)?(?:less\s+(?:than\s+)?or\s+equal\s+to|less\s+equal(?:\s+to)?|LESS\s+EQUAL(?:\s+TO)?|<=)\s+['\"]?([^\s,'\"\n\]]+)['\"]?", "less_equal"),
        # "on or after" / "on or before" (date phrasing)
        (r"(?:date|time|scheduled)\s+(?:is\s+)?on\s+or\s+after\s+['\"]?([^\s,'\"\n\]]+)['\"]?", "greater_equal"),
        (r"(?:date|time|scheduled)\s+(?:is\s+)?on\s+or\s+before\s+['\"]?([^\s,'\"\n\]]+)['\"]?", "less_equal"),
        # "AFTER 'date'" / "BEFORE 'date'" in date context
        (_FLD + r"\s+(?:is\s+)?(?:on\s+or\s+)?AFTER\s+['\"]([^'\"]+)['\"]", "greater_equal"),
        (_FLD + r"\s+(?:is\s+)?(?:on\s+or\s+)?BEFORE\s+['\"]([^'\"]+)['\"]", "less_equal"),
        # plain greater/less than
        (_FLD + r"\s+greater\s+than\s+['\"]?([^\s,'\"\n\]]+)['\"]?", "greater_than"),
        (_FLD + r"\s+less\s+than\s+['\"]?([^\s,'\"\n\]]+)['\"]?", "less_than"),
        # BELOW / ABOVE
        (_FLD + r"\s+BELOW\s+['\"]?([^\s,'\"\n\]]+)['\"]?", "less_than"),
        (_FLD + r"\s+ABOVE\s+['\"]?([^\s,'\"\n\]]+)['\"]?", "greater_than"),
    ]

    seen: set = set()
    for pat, op in basic:
        for m in re.finditer(pat, t, re.IGNORECASE):
            if _in_skip(m.start(), m.end()):
                continue
            field = m.group(1).strip().lower().replace(" ", "_")
            value = m.group(2).strip().strip("'\"").rstrip(".,;:")
            key = (field, op, value)
            if key not in seen:
                seen.add(key)
                constraints.append({"field": field, "op": op, "value": value})

    # Special: "value that is NOT 'X'" (destination-style)
    m2 = re.search(r"(\w+)\s+value\s+that\s+is\s+NOT\s+['\"]([^'\"]+)['\"]", t, re.IGNORECASE)
    if m2:
        field = m2.group(1).strip().lower()
        val = m2.group(2)
        key2 = (field, "not_equals", val)
        if key2 not in seen:
            seen.add(key2)
            constraints.append({"field": field, "op": "not_equals", "value": val})

    return constraints


def _format_constraints_block(constraints: List[Dict[str, Any]]) -> str:
    """Format parsed constraints for LLM context."""
    if not constraints:
        return ""
    lines = ["TASK_CONSTRAINTS (use these to find the RIGHT item and fill forms correctly):"]
    for c in constraints:
        field = c["field"]
        op = c["op"]
        value = c["value"]
        if op == "equals":
            lines.append(f"  [{field}] MUST EQUAL '{value}' exactly -> type/select this exact value")
        elif op == "not_equals":
            lines.append(f"  [{field}] must NOT be '{value}' -> choose any other valid value")
        elif op == "contains":
            lines.append(f"  [{field}] MUST CONTAIN the substring '{value}'")
        elif op == "not_contains":
            lines.append(f"  [{field}] must NOT contain '{value}'")
        elif op == "greater_than":
            lines.append(f"  [{field}] must be > {value} (numeric)")
        elif op == "less_than":
            lines.append(f"  [{field}] must be < {value} (numeric or date)")
        elif op == "greater_equal":
            lines.append(f"  [{field}] must be >= {value} (on or after / numeric)")
        elif op == "less_equal":
            lines.append(f"  [{field}] must be <= {value} (on or before / numeric)")
        elif op == "not_in":
            lines.append(f"  [{field}] must NOT be any of {value}")
        elif op == "in":
            lines.append(f"  [{field}] must be one of {value}")
    return "\n".join(lines)


def _classify_task(task: str) -> str:
    """Return a short label for the task type to prime the LLM."""
    t = (task or "").lower()

    # --- IWA-specific tasks (check first, most specific) ---

    # ---- AutoRide (8012) ----
    if re.search(r"(enter|type)\s+destination", t, re.IGNORECASE):
        return "ENTER_DESTINATION"
    if re.search(r"destination\s+(value\s+)?that\s+is\s+NOT", t, re.IGNORECASE):
        return "ENTER_DESTINATION"
    if re.search(r"enter\s+(and\s+select\s+)?a\s+location", t, re.IGNORECASE):
        return "ENTER_LOCATION"
    if re.search(r"location\s+equals\s+['\"]", t, re.IGNORECASE):
        return "ENTER_LOCATION"
    if re.search(r"search\s+ride\s+(details\s+)?where\s+the\s+location", t, re.IGNORECASE):
        return "SEARCH_RIDE"
    if re.search(r"(search|search\s+for)\s+.*location\s+.*destination", t, re.IGNORECASE):
        return "SEARCH_LOCATION"
    if re.search(r"search\s+location\s+(details|to\s+find|for)", t, re.IGNORECASE):
        return "SEARCH_LOCATION"
    if re.search(r"destination\s+equals\s+", t, re.IGNORECASE):
        return "SEARCH_LOCATION"
    if re.search(r"(reserve|book)\s+.*ride", t, re.IGNORECASE):
        return "RESERVE_RIDE"
    if re.search(r"cancel\s+reservation", t, re.IGNORECASE):
        return "CANCEL_RESERVATION"
    if re.search(r"select\s+(a\s+)?date\s+for\s+(the\s+|your\s+)?trip", t, re.IGNORECASE):
        return "SELECT_DATE"
    if re.search(r"select\s+(a\s+)?time\s+for\s+(my\s+|your\s+)?trip", t, re.IGNORECASE):
        return "SELECT_TIME"
    if re.search(r"select\s+time\s+for\s+my\s+trip", t, re.IGNORECASE):
        return "SELECT_TIME"
    if re.search(r"next\s+pickup", t, re.IGNORECASE):
        return "NEXT_PICKUP"

    # ---- AutoMail (8005) ----
    if re.search(r"mark\s+as\s+spam", t, re.IGNORECASE):
        return "MARK_AS_SPAM"
    if re.search(r"(mark|move)\s+.*(spam|junk)", t, re.IGNORECASE):
        return "MARK_AS_SPAM"
    if re.search(r"star\s+the\s+email", t, re.IGNORECASE):
        return "STAR_AN_EMAIL"
    if re.search(r"archive\s+the\s+email", t, re.IGNORECASE):
        return "ARCHIVE_EMAIL"
    if re.search(r"delete\s+the\s+email", t, re.IGNORECASE):
        return "DELETE_EMAIL"
    if re.search(r"forward\s+the\s+email", t, re.IGNORECASE):
        return "FORWARD_EMAIL"
    if re.search(r"mark.*email.*important|mark.*important.*email", t, re.IGNORECASE):
        return "MARK_EMAIL_AS_IMPORTANT"
    if re.search(r"mark\s+(the\s+)?email\s+as\s+unread", t, re.IGNORECASE):
        return "MARK_AS_UNREAD"
    if re.search(r"view\s+the\s+email\s+where", t, re.IGNORECASE):
        return "VIEW_EMAIL"
    if re.search(r"change\s+the\s+application\s+theme", t, re.IGNORECASE):
        return "THEME_CHANGED"
    if re.search(r"edit.*draft.*email", t, re.IGNORECASE):
        return "EDIT_DRAFT_EMAIL"
    if re.search(r"(next|go\s+to\s+the\s+next)\s+page\s+of\s+emails", t, re.IGNORECASE):
        return "EMAILS_NEXT_PAGE"
    if re.search(r"(previous|go\s+back\s+to\s+the\s+previous)\s+page\s+of\s+emails", t, re.IGNORECASE):
        return "EMAILS_PREV_PAGE"
    if re.search(r"(clear|deselect)\s+all\s+selected\s+emails", t, re.IGNORECASE):
        return "CLEAR_SELECTION"
    if re.search(r"send\s+.*using\s+the\s+template", t, re.IGNORECASE):
        return "TEMPLATE_SENT"
    if re.search(r"send\s+an\s+email\s+using\s+the\s+template", t, re.IGNORECASE):
        return "TEMPLATE_SENT"
    if re.search(r"save.*template.*draft", t, re.IGNORECASE):
        return "TEMPLATE_SAVED_DRAFT"
    if re.search(r"select\s+the\s+template", t, re.IGNORECASE):
        return "TEMPLATE_SELECTED"

    # ---- AutoCalendar (8010) ----
    if re.search(r"switch\s+to\s+week\s+view", t, re.IGNORECASE):
        return "SELECT_WEEK"
    if re.search(r"switch\s+to\s+month\s+view", t, re.IGNORECASE):
        return "SELECT_MONTH"
    if re.search(r"switch\s+to\s+day\s+view", t, re.IGNORECASE):
        return "SELECT_DAY"
    if re.search(r"switch\s+to\s+5.?day\s+view", t, re.IGNORECASE):
        return "SELECT_FIVE_DAYS"
    if re.search(r"(add\s+|click.*)\s*add\s+calendar\s+button", t, re.IGNORECASE):
        return "ADD_NEW_CALENDAR"
    if re.search(r"create\s+a\s+new\s+calendar", t, re.IGNORECASE):
        return "CREATE_CALENDAR"
    if re.search(r"add\s+an?\s+attendee\s+to\s+the\s+event", t, re.IGNORECASE):
        return "EVENT_ADD_ATTENDEE"
    if re.search(r"remove\s+an?\s+attendee\s+from\s+the\s+event", t, re.IGNORECASE):
        return "EVENT_REMOVE_ATTENDEE"
    if re.search(r"delete\s+an?\s+added\s+event", t, re.IGNORECASE):
        return "DELETE_ADDED_EVENT"
    if re.search(r"cancel\s+an?\s+event", t, re.IGNORECASE):
        return "CANCEL_ADD_EVENT"
    if re.search(r"open\s+the\s+event\s+creation\s+wizard", t, re.IGNORECASE):
        return "EVENT_WIZARD_OPEN"
    if re.search(r"click\s+on\s+cell\s+for\s+a\s+date", t, re.IGNORECASE):
        return "CELL_CLICKED"
    if re.search(r"click.*cell.*in\s+the\s+5\s+days\s+view", t, re.IGNORECASE):
        return "CELL_CLICKED"
    if re.search(r"add\s+a\s+new\s+calendar\s+event", t, re.IGNORECASE):
        return "NEW_CALENDAR_EVENT_ADDED"
    if re.search(r"add\s+an?\s+event\b", t, re.IGNORECASE):
        return "ADD_EVENT"
    if re.search(r"(show|view)\s+.*pending\s+events", t, re.IGNORECASE):
        return "VIEW_PENDING_EVENTS"
    if re.search(r"show\s+me\s+results\s+for\s+a\s+search\s+query", t, re.IGNORECASE):
        return "SEARCH_SUBMIT"

    # ---- AutoList (8011) ----
    if re.search(r"add\s+members?\s+to\s+the\s+team", t, re.IGNORECASE):
        return "AUTOLIST_TEAM_MEMBERS_ADDED"
    if re.search(r"assign\s+a\s+role\s+.*team\s+member", t, re.IGNORECASE):
        return "AUTOLIST_TEAM_ROLE_ASSIGNED"
    if re.search(r"edit\s+task\s+modal\s+open", t, re.IGNORECASE):
        return "AUTOLIST_EDIT_TASK_MODAL_OPENED"
    if re.search(r"button\s+to\s+add\s+a\s+task\s+is\s+clicked", t, re.IGNORECASE):
        return "AUTOLIST_ADD_TASK_CLICKED"
    if re.search(r"change\s+the\s+priority\s+to", t, re.IGNORECASE):
        return "AUTOLIST_SELECT_TASK_PRIORITY"
    if re.search(r"cancel\s+creating\s+the\s+task", t, re.IGNORECASE):
        return "AUTOLIST_CANCEL_TASK_CREATION"
    if re.search(r"create\s+a\s+team\s+whose", t, re.IGNORECASE):
        return "AUTOLIST_TEAM_CREATED"
    if re.search(r"delete\s+task\s+whose", t, re.IGNORECASE):
        return "AUTOLIST_DELETE_TASK"
    if re.search(r"add\s+a\s+task\s+whose", t, re.IGNORECASE):
        return "AUTOLIST_TASK_ADDED"
    if re.search(r"add\s+a\s+task\s+where", t, re.IGNORECASE):
        return "AUTOLIST_TASK_ADDED"

    # ---- AutoMedic (8013) ----
    if re.search(r"(show|retrieve)\s+details\s+(for\s+a\s+doctor|of\s+the\s+doctor\s+education|of\s+a\s+doctor)", t, re.IGNORECASE):
        if re.search(r"education|certif", t, re.IGNORECASE):
            return "VIEW_DOCTOR_EDUCATION"
        if re.search(r"availab", t, re.IGNORECASE):
            return "VIEW_DOCTOR_AVAILABILITY"
        return "VIEW_DOCTOR_PROFILE"
    if re.search(r"show\s+details\s+for\s+a\s+doctor", t, re.IGNORECASE):
        return "VIEW_DOCTOR_PROFILE"
    if re.search(r"retrieve\s+details\s+of\s+the\s+doctor\s+education", t, re.IGNORECASE):
        return "VIEW_DOCTOR_EDUCATION"
    if re.search(r"show\s+me\s+the\s+availability\s+details\s+for\s+a\s+doctor", t, re.IGNORECASE):
        return "VIEW_DOCTOR_AVAILABILITY"
    if re.search(r"show\s+me\s+(details\s+about\s+)?doctors", t, re.IGNORECASE):
        return "SEARCH_DOCTORS"
    if re.search(r"(search|retrieve)\s+(medical|details\s+of\s+medical)", t, re.IGNORECASE):
        return "SEARCH_MEDICAL_ANALYSIS"
    if re.search(r"view\s+medical\s+analysis", t, re.IGNORECASE):
        return "VIEW_MEDICAL_ANALYSIS"
    if re.search(r"open\s+appointment\s+form", t, re.IGNORECASE):
        return "OPEN_APPOINTMENT_FORM"
    if re.search(r"open\s+contact\s+doctor\s+form", t, re.IGNORECASE):
        return "OPEN_CONTACT_DOCTOR_FORM"
    if re.search(r"contact\s+a\s+doctor\s+where", t, re.IGNORECASE):
        return "DOCTOR_CONTACTED_SUCCESSFULLY"
    if re.search(r"contact\s+(a\s+)?doctor", t, re.IGNORECASE):
        return "CONTACT_DOCTOR"
    if re.search(r"retrieve\s+details\s+of\s+appointments", t, re.IGNORECASE):
        return "SEARCH_APPOINTMENT"
    if re.search(r"request\s+a\s+quick\s+appointment", t, re.IGNORECASE):
        return "REQUEST_QUICK_APPOINTMENT"
    if re.search(r"doctor.*education|education.*doctor", t, re.IGNORECASE):
        return "VIEW_DOCTOR_EDUCATION"

    # ---- AutoConnect (8008) ----
    if re.search(r"comment\s+on\s+the\s+post", t, re.IGNORECASE):
        return "COMMENT_ON_POST"
    if re.search(r"save\s+the\s+post\s+where", t, re.IGNORECASE):
        return "SAVE_POST"
    if re.search(r"follow\s+the\s+company\s+page", t, re.IGNORECASE):
        return "FOLLOW_PAGE"
    if re.search(r"unfollow\s+the\s+company\s+page", t, re.IGNORECASE):
        return "UNFOLLOW_PAGE"
    if re.search(r"(withdraw|cancel)\s+application", t, re.IGNORECASE):
        return "CANCEL_APPLICATION"
    if re.search(r"(search\s+for|show\s+me)\s+users", t, re.IGNORECASE):
        return "SEARCH_USERS"
    if re.search(r"go\s+back\s+to\s+all\s+jobs", t, re.IGNORECASE):
        return "BACK_TO_ALL_JOBS"
    if re.search(r"navigate\s+to\s+the\s+'?home'?\s+tab", t, re.IGNORECASE):
        return "HOME_NAVBAR"
    if re.search(r"show\s+me\s+my\s+hidden\s+posts", t, re.IGNORECASE):
        return "VIEW_HIDDEN_POSTS"
    if re.search(r"search\s+for\s+jobs\s+where\s+the\s+query", t, re.IGNORECASE):
        return "SEARCH_JOBS"
    if re.search(r"apply\s+for\s+(a\s+)?job", t, re.IGNORECASE):
        return "APPLY_FOR_JOB"
    if re.search(r"edit\s+profile\s+to\s+set\s+the\s+bio", t, re.IGNORECASE):
        return "EDIT_PROFILE_BIO"

    # ---- AutoHire (8009) ----
    if re.search(r"decide\s+to\s+remove\s+expert\s+from\s+hire\s+later", t, re.IGNORECASE):
        return "HIRE_LATER_REMOVED"
    if re.search(r"decide\s+to\s+hire\s+later", t, re.IGNORECASE):
        return "HIRE_LATER"
    if re.search(r"hire\s+(a\s+)?(consultant|expert|later)", t, re.IGNORECASE):
        if "later" in t:
            return "HIRE_LATER"
        return "HIRE_BTN_CLICKED"
    if re.search(r"show\s+me\s+details\s+about\s+a\s+hiring\s+team", t, re.IGNORECASE):
        return "SELECT_HIRING_TEAM"
    if re.search(r"select\s+a\s+project\s+size", t, re.IGNORECASE):
        return "CHOOSE_PROJECT_SIZE"
    if re.search(r"closing\s+the\s+job\s+posting\s+window", t, re.IGNORECASE):
        return "CLOSE_POST_A_JOB_WINDOW"
    if re.search(r"clicks?\s+on\s+the\s+jobs?\s+option\s+in\s+the\s+navbar", t, re.IGNORECASE):
        return "NAVBAR_JOBS_CLICK"
    if re.search(r"clicks?\s+on\s+.?hires?.?\s+from\s+the\s+navbar", t, re.IGNORECASE):
        return "NAVBAR_HIRES_CLICK"
    if re.search(r"searches?\s+for\s+a\s+skill", t, re.IGNORECASE):
        return "SEARCH_SKILL"
    if re.search(r"(job\s+posting|writing\s+(a\s+)?(strong\s+)?title\s+of\s+(the\s+)?job)", t, re.IGNORECASE):
        return "WRITE_JOB_TITLE"
    if re.search(r"edit\s+profile\s+about", t, re.IGNORECASE):
        return "EDIT_ABOUT"
    if re.search(r"update\s+my\s+profile\s+about\s+section", t, re.IGNORECASE):
        return "EDIT_ABOUT"
    if re.search(r"edit\s+profile\s+(location|email)", t, re.IGNORECASE):
        if "location" in t:
            return "EDIT_PROFILE_LOCATION"
        return "EDIT_PROFILE_EMAIL"
    if re.search(r"edit\s+profile\s+email", t, re.IGNORECASE):
        return "EDIT_PROFILE_EMAIL"

    # ---- AutoLodge (8007) ----
    if re.search(r"confirm\s+the\s+booking", t, re.IGNORECASE):
        return "BOOKING_CONFIRM"
    if re.search(r"(adjust|set|change)\s+the\s+number\s+of\s+guests", t, re.IGNORECASE):
        return "EDIT_NUMBER_OF_GUESTS"
    if re.search(r"(open\s+)?guest\s+selector\s+dropdown", t, re.IGNORECASE):
        return "PEOPLE_DROPDOWN_OPENED"
    if re.search(r"select\s+(a\s+)?payment\s+method", t, re.IGNORECASE):
        return "PAYMENT_METHOD_SELECTED"
    if re.search(r"(reserve|book)\s+the\s+hotel", t, re.IGNORECASE):
        return "RESERVE_HOTEL"
    if re.search(r"share\s+the\s+hotel\s+listing", t, re.IGNORECASE):
        return "SHARE_HOTEL"
    if re.search(r"show\s+(me\s+)?details\s+for\s+popular\s+hotels", t, re.IGNORECASE):
        return "POPULAR_HOTELS_VIEWED"
    if re.search(r"search\s+for\s+hotels?", t, re.IGNORECASE):
        return "SEARCH_HOTEL"
    if re.search(r"submit\s+a\s+review\b(?!.*restaurant)", t, re.IGNORECASE):
        return "SUBMIT_REVIEW"
    if re.search(r"add\s+to\s+wishlist.*hotel", t, re.IGNORECASE):
        return "ADD_TO_WISHLIST_HOTEL"
    if re.search(r"apply.*filter.*hotel|show\s+details\s+for\s+hotels", t, re.IGNORECASE):
        return "APPLY_FILTERS"

    # ---- AutoDelivery (8006) ----
    if re.search(r"(next|show\s+me\s+the\s+next)\s+set\s+of\s+restaurants", t, re.IGNORECASE):
        return "RESTAURANT_NEXT_PAGE"
    if re.search(r"go\s+back\s+to\s+the\s+previous\s+page\s+of\s+restaurants", t, re.IGNORECASE):
        return "RESTAURANT_PREV_PAGE"
    if re.search(r"return\s+to\s+all\s+restaurants", t, re.IGNORECASE):
        return "BACK_TO_ALL_RESTAURANTS"
    if re.search(r"increase\s+the\s+quantity\s+of\s+the\s+item\s+in\s+the\s+cart", t, re.IGNORECASE):
        return "ITEM_INCREMENTED"
    if re.search(r"search\s+for\s+restaurants?\s+(where|that)", t, re.IGNORECASE):
        return "SEARCH_DELIVERY_RESTAURANT"
    if re.search(r"submit\s+(a\s+)?review\s+for\s+(a\s+)?restaurant", t, re.IGNORECASE):
        return "REVIEW_SUBMITTED"
    if re.search(r"add\s+an?\s+address\s+that\s+is", t, re.IGNORECASE):
        return "ADDRESS_ADDED"
    if re.search(r"set\s+dropoff\s+preference", t, re.IGNORECASE):
        return "DROPOFF_PREFERENCE"
    if re.search(r"select\s+(a\s+)?delivery\s+priority", t, re.IGNORECASE):
        return "DELIVERY_PRIORITY_SELECTED"
    if re.search(r"view\s+the\s+details\s+of\s+a\s+restaurant\s+where", t, re.IGNORECASE):
        return "VIEW_DELIVERY_RESTAURANT"
    if re.search(r"show\s+all\s+restaurants", t, re.IGNORECASE):
        return "VIEW_ALL_RESTAURANTS"
    if re.search(r"(go\s+to\s+)?checkout\s+and\s+show\s+the\s+order", t, re.IGNORECASE):
        return "OPEN_CHECKOUT_PAGE"

    # ---- AutoRestaurant (8003) ----
    if re.search(r"search\s+for\s+restaurants?\s+where\s+the\s+query", t, re.IGNORECASE):
        return "SEARCH_RESTAURANT"
    if re.search(r"please\s+collapse\s+the\s+menu", t, re.IGNORECASE):
        return "COLLAPSE_MENU"
    if re.search(r"click\s+the\s+contact\s+card\s+where", t, re.IGNORECASE):
        return "CONTACT_CARD_CLICK"
    if re.search(r"scroll\s+in\s+the\s+direction", t, re.IGNORECASE):
        return "SCROLL_VIEW"
    if re.search(r"show\s+details\s+for\s+the\s+help\s+category", t, re.IGNORECASE):
        return "HELP_CATEGORY_SELECTED"
    if re.search(r"(navigate\s+to|find)\s+the\s+help\s+page", t, re.IGNORECASE):
        return "HELP_PAGE_VIEW"
    if re.search(r"(open|show).*guest.*selector.*dropdown.*number\s+of\s+people", t, re.IGNORECASE):
        return "PEOPLE_DROPDOWN_OPENED"
    if re.search(r"select.*country.*dropdown|please\s+select\s+the\s+country", t, re.IGNORECASE):
        return "COUNTRY_SELECTED"
    if re.search(r"expand\s+the\s+faq\s+item", t, re.IGNORECASE):
        return "HELP_FAQ_TOGGLED"
    if re.search(r"open\s+the\s+help", t, re.IGNORECASE):
        return "HELP_VIEWED"
    if re.search(r"click\s+on\s+the\s+feature.*on\s+the\s+about\s+page", t, re.IGNORECASE):
        return "ABOUT_FEATURE_CLICK"
    if re.search(r"contact\s+support\s+regarding", t, re.IGNORECASE):
        return "CONTACT_FORM_SUBMIT"
    if re.search(r"view\s+the\s+details\s+of\s+a\s+restaurant", t, re.IGNORECASE):
        return "VIEW_RESTAURANT"
    if re.search(r"show\s+details\s+for\s+a\s+restaurant", t, re.IGNORECASE):
        return "VIEW_RESTAURANT"

    # ---- AutoShop (8002) ----
    if re.search(r"update\s+quantity\s+of\s+item\s+with\s+title", t, re.IGNORECASE):
        return "QUANTITY_CHANGED"
    if re.search(r"update\s+the\s+quantity\s+of\s+the\s+item\s+in\s+my\s+cart", t, re.IGNORECASE):
        return "QUANTITY_CHANGED"
    if re.search(r"update\s+quantity\s+of\s+item", t, re.IGNORECASE):
        return "QUANTITY_CHANGED"
    if re.search(r"increase\s+the\s+quantity", t, re.IGNORECASE):
        return "ITEM_INCREMENTED"
    if re.search(r"show\s+details\s+for\s+a\s+product", t, re.IGNORECASE):
        return "VIEW_DETAIL"
    if re.search(r"filter\s+to\s+show\s+only\s+products\s+in\s+the\s+category", t, re.IGNORECASE):
        return "CATEGORY_FILTER"
    if re.search(r"(show\s+me\s+my\s+saved\s+items|my\s+wishlist|show.*wishlist)", t, re.IGNORECASE):
        return "VIEW_WISHLIST"
    if re.search(r"proceed\s+to\s+checkout", t, re.IGNORECASE):
        return "PROCEED_TO_CHECKOUT"
    if re.search(r"(complete\s+my\s+purchase|complete\s+my\s+order)", t, re.IGNORECASE):
        return "ORDER_COMPLETED"
    if re.search(r"scroll\s+(left|right)\s+in\s+the\s+carousel", t, re.IGNORECASE):
        return "CAROUSEL_SCROLL"
    if re.search(r"share\s+the\s+link\s+to\s+a\s+product", t, re.IGNORECASE):
        return "SHARE_PRODUCT"
    if re.search(r"add.*this.*item.*to.*cart", t, re.IGNORECASE):
        return "ADD_TO_CART"
    if re.search(r"(add|put).*wishlist\s+(a\s+)?(?:hotel|item|product|book)", t, re.IGNORECASE):
        return "ADD_TO_WISHLIST"
    if re.search(r"(show|view)\s+my\s+shopping\s+cart", t, re.IGNORECASE):
        return "VIEW_CART"

    # ---- AutoDoc (8004) ----
    if re.search(r"add\s+a\s+new\s+client", t, re.IGNORECASE):
        return "ADD_CLIENT"
    if re.search(r"add\s+a\s+new\s+matter", t, re.IGNORECASE):
        return "ADD_NEW_MATTER"
    if re.search(r"search\s+for\s+matters?\s+where\s+the\s+query", t, re.IGNORECASE):
        return "SEARCH_MATTER"
    if re.search(r"show\s+me\s+details\s+for\s+clients?\s+whose", t, re.IGNORECASE):
        return "FILTER_CLIENTS"
    if re.search(r"show\s+me\s+matters?\s+where\s+the\s+status", t, re.IGNORECASE):
        return "FILTER_MATTER_STATUS"
    if re.search(r"show\s+me\s+details\s+about\s+a\s+document", t, re.IGNORECASE):
        return "DOCUMENT_DELETED"
    if re.search(r"sort\s+matters?\s+so\s+that", t, re.IGNORECASE):
        return "SORT_MATTER_BY_CREATED_AT"
    if re.search(r"change\s+(user\s+)?name\s+to", t, re.IGNORECASE):
        return "CHANGE_USER_NAME"
    if re.search(r"show.*pending\s+events\s+on\s+the\s+calendar", t, re.IGNORECASE):
        return "VIEW_PENDING_EVENTS"
    if re.search(r"add\s+a\s+new\s+calendar\s+event\s+where", t, re.IGNORECASE):
        return "NEW_CALENDAR_EVENT_ADDED"

    # ---- AutoBooks (8001) ----
    if re.search(r"(delete|remove)\s+(your\s+)?(user[- _]?registered\s+)?book", t, re.IGNORECASE):
        if re.search(r"\b(login|sign.?in)\b", t, re.IGNORECASE):
            return "DELETE_BOOK"
    if re.search(r"modify\s+your\s+book|edit\s+(your\s+)?book\s+where", t, re.IGNORECASE):
        return "EDIT_BOOK"
    if re.search(r"remove\s+from\s+the\s+reading\s+list", t, re.IGNORECASE):
        return "REMOVE_FROM_READING_LIST"
    if re.search(r"go\s+to\s+the\s+contact\s+page\s+and\s+send", t, re.IGNORECASE):
        return "CONTACT_BOOK"
    if re.search(r"register\s+with\s+the\s+following\s+username", t, re.IGNORECASE):
        return "REGISTRATION_BOOK"
    if re.search(r"show\s+details\s+for\s+a\s+book\s+where", t, re.IGNORECASE):
        return "BOOK_DETAIL"
    if re.search(r"filter\s+books?\s+where", t, re.IGNORECASE):
        return "FILTER_BOOK"
    if re.search(r"search\s+for\s+(the\s+)?book\s+with\s+the\s+query", t, re.IGNORECASE):
        return "SEARCH_BOOK"
    if re.search(r"view\s+the\s+shopping\s+cart.*all\s+items|see\s+all\s+items.*cart", t, re.IGNORECASE):
        return "VIEW_CART_BOOK"
    if re.search(r"login\s+for\s+the\s+following\s+username", t, re.IGNORECASE):
        return "LOGIN_BOOK"
    if re.search(r"authenticate\s+with\s+username.*view\s+the\s+shopping\s+cart", t, re.IGNORECASE):
        return "VIEW_CART_BOOK"

    # ---- AutoCinema (8000) specific ----
    if re.search(r"add\s+(to\s+)?watchlist", t, re.IGNORECASE):
        return "ADD_TO_WATCHLIST"
    if re.search(r"remove\s+from\s+watchlist", t, re.IGNORECASE):
        return "REMOVE_FROM_WATCHLIST"
    if re.search(r"share\s+movie\s+details", t, re.IGNORECASE):
        return "SHARE_MOVIE"
    if re.search(r"watch\s+the\s+trailer\s+for\s+a\s+movie", t, re.IGNORECASE):
        return "WATCH_TRAILER"

    # ---- AutoShop (8002) additional ----
    if re.search(r"click\s+on\s+buy\s+now\s+to\s+initiate\s+checkout", t, re.IGNORECASE):
        return "CHECKOUT_STARTED"

    # ---- AutoRestaurant (8003) additional ----
    if re.search(r"navigate\s+to\s+the\s+about\s+page", t, re.IGNORECASE):
        return "ABOUT_PAGE_VIEW"
    if re.search(r"open\s+the\s+date\s+selector", t, re.IGNORECASE):
        return "DATE_DROPDOWN_OPENED"
    if re.search(r"(retrieve\s+details\s+of\s+a\s+contact\s+form|submit.*contact.*form.*email.*contains)", t, re.IGNORECASE):
        return "CONTACT_FORM_SUBMIT"

    # ---- AutoDoc (8004) additional ----
    if re.search(r"edit\s+log\s+entry\s+where", t, re.IGNORECASE):
        return "LOG_EDITED"
    if re.search(r"archive\s+the\s+matter\s+where", t, re.IGNORECASE):
        return "ARCHIVE_MATTER"
    if re.search(r"(retrieve|show)\s+details\s+(of|for)\s+a?\s*client\s+whose", t, re.IGNORECASE):
        return "VIEW_CLIENT_DETAILS"
    if re.search(r"(retrieve|show)\s+details\s+(of|for)\s+(the\s+)?matter\s+(whose|where)", t, re.IGNORECASE):
        return "VIEW_MATTER_DETAILS"

    # ---- AutoMail (8005) additional ----
    if re.search(r"send\s+an\s+email\s+to\s+['\"]", t, re.IGNORECASE):
        return "SEND_EMAIL"
    if re.search(r"search\s+for\s+emails?\s+where\s+the\s+query", t, re.IGNORECASE):
        return "SEARCH_EMAIL"

    # ---- AutoDelivery (8006) additional ----
    if re.search(r"show\s+me\s+restaurants?\s+that\s+do\s+NOT", t, re.IGNORECASE):
        return "RESTAURANT_FILTER"
    if re.search(r"add\s+a?\s*menu\s+item\s+to\s+(my\s+)?cart", t, re.IGNORECASE):
        return "ADD_TO_CART_MENU_ITEM"
    if re.search(r"open\s+the\s+add.?to.?cart\s+modal", t, re.IGNORECASE):
        return "ADD_TO_CART_MODAL_OPEN"
    if re.search(r"start\s+a\s+quick\s+order", t, re.IGNORECASE):
        return "QUICK_ORDER_STARTED"

    # ---- AutoLodge (8007) additional ----
    if re.search(r"message\s+the\s+host\s+where", t, re.IGNORECASE):
        return "MESSAGE_HOST"
    if re.search(r"edit\s+check.?in.*check.?out\s+dates", t, re.IGNORECASE):
        return "EDIT_CHECK_IN_OUT_DATES"
    if re.search(r"open\s+my\s+wishlist\s+to\s+view\s+saved\s+hotels", t, re.IGNORECASE):
        return "WISHLIST_OPENED"

    # ---- AutoConnect (8008) additional ----
    if re.search(r"edit\s+profile\s+information", t, re.IGNORECASE):
        return "EDIT_PROFILE"
    if re.search(r"post\s+a\s+status\s+update", t, re.IGNORECASE):
        return "POST_STATUS"

    # ---- AutoHire (8009) additional ----
    if re.search(r"clicks?\s+the\s+'?experts?'?\s+option\s+in\s+the\s+navbar|list\s+of\s+all\s+experts.*clicks?\s+the\s+'?experts?", t, re.IGNORECASE):
        return "NAVBAR_EXPERTS_CLICK"
    if re.search(r"show\s+the\s+list\s+of\s+all\s+experts", t, re.IGNORECASE):
        return "NAVBAR_EXPERTS_CLICK"
    if re.search(r"add\s+a\s+skill\s+where\s+skill", t, re.IGNORECASE):
        return "ADD_SKILL"
    if re.search(r"submit\s+a\s+job\s+with\s+title", t, re.IGNORECASE):
        return "SUBMIT_JOB"
    if re.search(r"decide\s+to\s+start\s+hiring", t, re.IGNORECASE):
        return "HIRE_LATER_START"

    # ---- AutoCalendar (8010) additional ----
    if re.search(r"select\s+the\s+calendar\s+that\s+contains", t, re.IGNORECASE):
        return "SELECT_CALENDAR"
    if re.search(r"unselect\s+the\s+calendar", t, re.IGNORECASE):
        return "UNSELECT_CALENDAR"
    if re.search(r"go\s+to\s+today'?s?\s+date\s+in\s+the\s+calendar", t, re.IGNORECASE):
        return "SELECT_TODAY"

    # ---- AutoList (8011) additional ----
    if re.search(r"complete\s+task\s+where\s+the\s+name\s+equals", t, re.IGNORECASE):
        return "AUTOLIST_COMPLETE_TASK"

    # ---- AutoRide (8012) additional ----
    if re.search(r"view\s+trip\s+details\s+for\s+(a\s+)?(trip|ride)\s+where", t, re.IGNORECASE):
        return "TRIP_DETAILS"
    if re.search(r"select\s+car\s+options\s+where", t, re.IGNORECASE):
        return "SELECT_CAR"
    if re.search(r"search\s+destination\s+where\s+the\s+destination", t, re.IGNORECASE):
        return "SEARCH_DESTINATION"
    if re.search(r"select\s+date\s+for\s+(your|my)\s+trip\s+as", t, re.IGNORECASE):
        return "SELECT_DATE"

    # ---- AutoMedic (8013) additional ----
    if re.search(r"refill\s+prescription\s+where", t, re.IGNORECASE):
        return "REFILL_PRESCRIPTION"
    if re.search(r"(show\s+me\s+details\s+to\s+refill|show\s+details\s+for\s+a\s+prescription)", t, re.IGNORECASE):
        return "VIEW_PRESCRIPTION"
    if re.search(r"show\s+details\s+for\s+doctor\s+reviews\s+where", t, re.IGNORECASE):
        return "FILTER_DOCTOR_REVIEWS"

    # ---- AutoBooks login/logout ----
    if re.search(r"(login\s+for\s+the\s+following|login\s+with\s+(a\s+)?specific).*username.*then\s+logout", t, re.IGNORECASE):
        return "LOGOUT_BOOK"
    if re.search(r"first.*authenticate.*username.*then.*logout", t, re.IGNORECASE):
        return "LOGOUT_BOOK"

    # ---- AutoCinema/AutoBooks multi-step ----
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
    if re.search(r"\b(purchase|buy|checkout|order)\b", t) and re.search(r"\b(login|sign.?in|authenticate)\b", t):
        return "LOGIN_THEN_PURCHASE"

    # ---- Task management ----
    if re.search(r"delete\s+task\b", t, re.IGNORECASE):
        return "DELETE_TASK"
    if re.search(r"(create|add|new)\s+task\b", t, re.IGNORECASE):
        return "CREATE_TASK"
    if re.search(r"(edit|update|modify)\s+task\b", t, re.IGNORECASE):
        return "EDIT_TASK"

    # ---- Single-step / generic ----
    if re.search(r"\b(register|sign.?up|create.*account|fill.*registration)\b", t):
        return "REGISTRATION"
    if re.search(r"\b(login|sign.?in|log.?in|fill.*login|authenticate)\b", t):
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
    # ---- AutoRide (8012) ----
    "SEARCH_LOCATION": (
        "PLAYBOOK: 1) Find the search/destination input field on the page. "
        "2) Click it to focus. 3) Type the destination EXACTLY as given in TASK_CONSTRAINTS. "
        "4) If a dropdown/suggestions appear, click the matching result. "
        "5) Submit/confirm if needed."
    ),
    "RESERVE_RIDE": (
        "PLAYBOOK: 1) Browse available rides listed on the page. "
        "2) Use list_cards to see all rides with their name, price, location, destination, scheduled time. "
        "3) Find the ride matching ALL TASK_CONSTRAINTS (check location/NOT, destination/NOT, ride_name/NOT, scheduled time). "
        "4) Click 'Reserve' on the matching ride."
    ),
    "CANCEL_RESERVATION": (
        "PLAYBOOK: 1) Navigate to reservations/upcoming rides page. "
        "2) Use list_cards to see all reservations. "
        "3) Find the reservation matching ALL TASK_CONSTRAINTS (location, destination, ride_name, scheduled time). "
        "4) Click 'Cancel' on the matching reservation. 5) Confirm if prompted."
    ),
    "SELECT_DATE": (
        "PLAYBOOK: 1) Find the date picker/calendar widget on the page. "
        "2) Click it to open. 3) Select a date that satisfies the TASK_CONSTRAINTS "
        "(e.g., 'on or after 2026-03-17' → pick that date or any future date). "
        "4) Confirm the selection."
    ),
    "SELECT_TIME": (
        "PLAYBOOK: 1) Find the time picker/dropdown. "
        "2) Click to open. 3) Select a time that satisfies the constraint "
        "(e.g., 'less than or equal to 18:20:00' → pick any time ≤ 18:20). "
        "4) Confirm."
    ),
    "NEXT_PICKUP": (
        "PLAYBOOK: 1) Look for a 'Next Pickup' or scheduled ride section on the page. "
        "2) Find the pickup that satisfies the date/time constraints. "
        "3) Click to view its details."
    ),
    # ---- AutoMail (8005) ----
    "STAR_AN_EMAIL": (
        "PLAYBOOK: 1) Browse the inbox email list. "
        "2) Find the email matching ALL constraints: subject contains X, from_email equals Y, is_starred = False. "
        "3) Click the Star icon (☆) on that email row. "
        "Note: is_starred=False means it is NOT currently starred - you need to star it."
    ),
    "ARCHIVE_EMAIL": (
        "PLAYBOOK: 1) Browse the inbox. "
        "2) Find email matching: from_email NOT equals X, subject CONTAINS Y. "
        "3) Click on that email. 4) Find Archive button (box with arrow icon). Click it."
    ),
    "DELETE_EMAIL": (
        "PLAYBOOK: 1) Find the email matching subject/from_email constraints. "
        "2) Click the Delete/Trash icon on that email row."
    ),
    "FORWARD_EMAIL": (
        "PLAYBOOK: 1) Find the email matching subject/body constraints. "
        "2) Click to open the email. 3) Click 'Forward' button. "
        "4) Fill in the 'To' field if needed. 5) Send."
    ),
    "MARK_EMAIL_AS_IMPORTANT": (
        "PLAYBOOK: 1) Find the email matching from_email/subject constraints. "
        "2) Click the Important/Flag icon on that email. "
        "Note: is_important=True may mean it should already be marked - check and mark if not."
    ),
    "EDIT_DRAFT_EMAIL": (
        "PLAYBOOK: 1) Navigate to Drafts folder. "
        "2) Find draft matching 'to equals X' and 'subject not contains Y'. "
        "3) Click to open/edit the draft."
    ),
    "EMAILS_NEXT_PAGE": (
        "PLAYBOOK: 1) Look at the bottom of the email list for pagination controls. "
        "2) Click the '>' or 'Next' arrow/button to go to the next page."
    ),
    "EMAILS_PREV_PAGE": (
        "PLAYBOOK: 1) Look for '<' or 'Previous' arrow at bottom of email list. "
        "2) Click it to go to the previous page."
    ),
    "CLEAR_SELECTION": (
        "PLAYBOOK: 1) Look for a 'Clear Selection' button, or uncheck the 'Select All' checkbox "
        "at the top of the email list. 2) Click it."
    ),
    "TEMPLATE_SENT": (
        "PLAYBOOK: 1) Navigate to Templates section. "
        "2) Find the template matching constraints (template_name NOT contains X, to NOT contains Y). "
        "3) Click 'Send' or 'Use Template' on that template."
    ),
    "TEMPLATE_SAVED_DRAFT": (
        "PLAYBOOK: 1) Navigate to Templates section. "
        "2) Find template matching constraints. "
        "3) Click 'Save as Draft' on it."
    ),
    "TEMPLATE_SELECTED": (
        "PLAYBOOK: 1) Navigate to Templates section. "
        "2) Find the template where template_name contains X and subject NOT contains Y. "
        "3) Click 'Select' or 'Use' on it."
    ),
    # ---- AutoCalendar (8010) ----
    "SELECT_WEEK": (
        "PLAYBOOK: 1) Find the view switcher buttons (Day/5-day/Week/Month). "
        "2) Click 'Week' button."
    ),
    "SELECT_MONTH": (
        "PLAYBOOK: 1) Find view buttons. 2) Click 'Month' button."
    ),
    "SELECT_FIVE_DAYS": (
        "PLAYBOOK: 1) Find view buttons. 2) Click '5-day' or 'Work Week' button."
    ),
    "ADD_NEW_CALENDAR": (
        "PLAYBOOK: 1) Find the '+' or 'Add Calendar' button in the left sidebar. "
        "2) Click it to open the modal."
    ),
    "CREATE_CALENDAR": (
        "PLAYBOOK: 1) Click the '+' button next to 'Other calendars' in the sidebar. "
        "2) Fill in name (satisfying NOT constraints) and description (satisfying constraints). "
        "3) Click Create/Save."
    ),
    "EVENT_ADD_ATTENDEE": (
        "PLAYBOOK: 1) Find an event on the calendar (any event is fine unless constraints specify which). "
        "2) Click on it to open. 3) Click Edit. 4) Find 'Add Attendee' or attendees email field. "
        "5) Type an email address that satisfies constraints (NOT contains X). 6) Save."
    ),
    "DELETE_ADDED_EVENT": (
        "PLAYBOOK: 1) Browse calendar events using list_cards or visible_text. "
        "2) Find the event matching ALL constraints (reminders, description, all_day, location, recurrence, title, date). "
        "3) Click on that event. 4) Click Delete. 5) Confirm."
    ),
    "CANCEL_ADD_EVENT": (
        "PLAYBOOK: 1) Find the event matching constraints (location, start_time/NOT, end_time/NOT, title/NOT, visibility/NOT, date). "
        "2) Click on it. 3) Click Cancel/Delete. 4) Confirm."
    ),
    "NEW_CALENDAR_EVENT_ADDED": (
        "PLAYBOOK: 1) Click the '+' or 'Add Event' button on the calendar. "
        "2) Fill in the event form: "
        "   - label: choose one that does NOT contain the excluded value "
        "   - time: use the given time (e.g., '8:00am') "
        "   - date: pick a date satisfying the constraint (e.g., <= '2026-04-23') "
        "   - event_type: must NOT contain excluded value (choose 'Other' or different type). "
        "3) Save the event."
    ),
    "ADD_EVENT": (
        "PLAYBOOK: 1) Click '+' or on a time slot to add event. "
        "2) Fill ALL fields from TASK_CONSTRAINTS: "
        "   - title: exact value (if equals) or any value not equal to excluded one "
        "   - visibility: if 'not equals Public' → choose Private or Default "
        "   - reminders: if 'not equals 60' → use any other value "
        "   - date, time, meeting_link, attendees as constrained. "
        "3) Save."
    ),
    "VIEW_PENDING_EVENTS": (
        "PLAYBOOK: 1) Switch to a view that shows upcoming/pending events. "
        "2) Find events matching constraint (earliest date NOT equals X). "
        "3) Navigate to or click on that event."
    ),
    # ---- AutoList (8011) ----
    "AUTOLIST_TEAM_MEMBERS_ADDED": (
        "PLAYBOOK: 1) Navigate to the Team section/tab. "
        "2) Click 'Add Member' or similar. "
        "3) Search for a member whose name does NOT include the excluded names, "
        "ensuring member_count will be <= the given limit. "
        "4) Add them. Repeat if multiple members needed."
    ),
    "AUTOLIST_TEAM_ROLE_ASSIGNED": (
        "PLAYBOOK: 1) Go to Team section. "
        "2) Find a member whose name does NOT contain the excluded string. "
        "3) Click their role dropdown. 4) Select a role containing 'r' (or as specified)."
    ),
    "AUTOLIST_EDIT_TASK_MODAL_OPENED": (
        "PLAYBOOK: 1) Browse task list. "
        "2) Find task matching ALL constraints (name contains X, description equals Y, date < Z, priority NOT equals W). "
        "3) Click the Edit/Pencil icon on that task to open the edit modal."
    ),
    "AUTOLIST_ADD_TASK_CLICKED": (
        "PLAYBOOK: 1) Find the 'Add Task' button ('+' or 'Add Task' text). 2) Click it."
    ),
    "AUTOLIST_TASK_ADDED": (
        "PLAYBOOK: 1) Click 'Add Task' button. "
        "2) Fill in name = EXACT value, description = EXACT value, "
        "date = satisfying constraint, priority = exact value. "
        "3) Save."
    ),
    "AUTOLIST_DELETE_TASK": (
        "PLAYBOOK: 1) On AutoList, navigate to the Tasks section. "
        "2) Use list_cards to see all tasks with fields: name, description, date, priority. "
        "3) Find the task matching ALL TASK_CONSTRAINTS: "
        "   - name NOT contains the excluded string "
        "   - description contains the required substring "
        "   - date less_than the given date "
        "   - priority equals the given value. "
        "4) Click that task's Delete/Trash/Remove button. "
        "5) Confirm deletion if a confirmation dialog appears."
    ),
    "CONFIRM_AND_PAY": (
        "PLAYBOOK: 1) You are on a lodging/accommodation booking site (AutoLodge). "
        "2) Use list_cards to browse listings. Find the one matching ALL TASK_CONSTRAINTS: "
        "   - guests_set: equals or NOT equals "
        "   - title: NOT equals "
        "   - price: less_equal or greater_than "
        "   - host_name: contains "
        "   - reviews: equals "
        "   - amenities: NOT in list "
        "   - rating: equals "
        "   - location: contains "
        "3) Click 'Book Now' or 'Reserve' on the matched listing. "
        "4) Fill the payment form with EXACT values from TASK_CONSTRAINTS: "
        "   - card_number: use a different card if NOT equals constraint "
        "   - expiration: use a different date if NOT equals constraint "
        "   - cvv: type EXACTLY as specified "
        "   - zipcode: type EXACTLY as specified "
        "   - country: type EXACTLY or choose from dropdown. "
        "5) Submit/Confirm the booking."
    ),
    # ---- AutoMedic (8013) ----
    "VIEW_DOCTOR_PROFILE": (
        "PLAYBOOK: 1) Browse doctor list using list_cards. "
        "2) Find doctor matching ALL constraints (doctor_name equals X, speciality NOT equals Y, rating equals Z, fee <= W, language NOT contains Q). "
        "3) Click on that doctor's card to view their profile/details."
    ),
    "SEARCH_DOCTORS": (
        "PLAYBOOK: 1) Find the search/filter fields for doctors. "
        "2) Enter search criteria matching constraints (speciality contains X, language contains Y). "
        "3) Submit search."
    ),
    "SEARCH_MEDICAL_ANALYSIS": (
        "PLAYBOOK: 1) Navigate to Medical Records/Analysis section. "
        "2) Use search/filter fields: enter record_title, doctor_name constraints. "
        "3) Submit/search."
    ),
    "VIEW_MEDICAL_ANALYSIS": (
        "PLAYBOOK: 1) Navigate to Medical Records. "
        "2) Find the record matching constraints. 3) Click to view details."
    ),
    "OPEN_APPOINTMENT_FORM": (
        "PLAYBOOK: 1) Browse doctor cards using list_cards. "
        "2) Find doctor matching ALL constraints (doctor_name contains X, speciality NOT contains Y, etc.). "
        "3) Click 'Book Appointment' on that doctor. "
        "4) Fill in date equals X and time equals Y from TASK_CONSTRAINTS. 5) Open/submit the form."
    ),
    "OPEN_CONTACT_DOCTOR_FORM": (
        "PLAYBOOK: 1) Find doctor matching ALL constraints (doctor_name NOT contains X, speciality NOT equals Y, rating equals Z, fee > W). "
        "2) Click 'Contact Doctor' button on that doctor's card. "
        "3) This opens the contact form - form should be open/visible."
    ),
    "CONTACT_DOCTOR": (
        "PLAYBOOK: 1) Find doctor matching constraints (doctor_name contains X, rating > Y, language contains Z, speciality NOT W, fee > Q). "
        "2) Click 'Contact' on their card. 3) Fill the contact form if needed. 4) Submit."
    ),
    "SEARCH_APPOINTMENT": (
        "PLAYBOOK: 1) Go to Appointments section. "
        "2) Search/filter for appointments where doctor_name NOT X, speciality contains Y, date equals Z. "
        "3) View results."
    ),
    "REQUEST_QUICK_APPOINTMENT": (
        "PLAYBOOK: 1) Find 'Quick Appointment' or 'Request Appointment' button. "
        "2) Fill form: speciality NOT X, patient_name NOT contains Y, patient_email equals Z. "
        "3) Submit."
    ),
    "VIEW_DOCTOR_EDUCATION": (
        "PLAYBOOK: 1) On AutoMedic, browse the doctors list. "
        "2) Find the doctor matching ALL TASK_CONSTRAINTS: "
        "   - doctor_name equals/contains/NOT contains "
        "   - speciality equals/NOT equals/contains "
        "   - rating equals/greater_than/NOT equals "
        "   - consultation_fee equals/less_equal "
        "   - language equals/NOT contains. "
        "3) Click on that doctor's card/row. "
        "4) On their profile page, find the 'Education' or 'Certifications' tab/section. "
        "5) Click it to reveal their education details."
    ),
    # ---- AutoConnect (8008) ----
    "COMMENT_ON_POST": (
        "PLAYBOOK: 1) Find a post in the feed (scroll if needed). "
        "2) Click the 'Comment' button/link on it. "
        "3) Type the EXACT comment text from TASK_CREDENTIALS or task text. "
        "4) Submit the comment."
    ),
    "FOLLOW_PAGE": (
        "PLAYBOOK: 1) Find the company page recommended by the named person, or matching constraints. "
        "2) Click the 'Follow' button on that company page."
    ),
    "UNFOLLOW_PAGE": (
        "PLAYBOOK: 1) Find the company page with the given name. "
        "2) Click 'Unfollow' button."
    ),
    "CANCEL_APPLICATION": (
        "PLAYBOOK: 1) Navigate to My Applications or Jobs section. "
        "2) Find the application where job_title equals X and status equals Y. "
        "3) Click 'Withdraw' or 'Cancel Application'."
    ),
    "SEARCH_USERS": (
        "PLAYBOOK: 1) Find the user search bar (People section or top search). "
        "2) Type the query containing the given substring. 3) Submit."
    ),
    "BACK_TO_ALL_JOBS": (
        "PLAYBOOK: 1) Navigate to Jobs section. "
        "2) Find a job where location equals X, company NOT Y, title NOT contains Z. "
        "3) Click on it. 4) Click 'Back to Jobs' breadcrumb/link."
    ),
    "EDIT_PROFILE_BIO": (
        "PLAYBOOK: 1) Navigate to Profile/Settings. "
        "2) Find Bio field. 3) Set bio to EXACT value from task. 4) Save."
    ),
    # ---- AutoHire (8009) ----
    "HIRE_BTN_CLICKED": (
        "PLAYBOOK: 1) Browse expert/consultant list. "
        "2) Find expert matching ALL constraints (role contains X, country equals Y, name contains Z). "
        "3) Click 'Hire Now' button on that expert."
    ),
    "HIRE_LATER": (
        "PLAYBOOK: 1) Browse expert list using list_cards. "
        "2) Find expert where name NOT contains X, role NOT Y, country contains Z. "
        "3) Click 'Hire Later' button on that expert."
    ),
    "HIRE_LATER_REMOVED": (
        "PLAYBOOK: 1) Navigate to 'Hire Later' saved page. "
        "2) Find expert where name NOT X, country contains Y, role contains Z. "
        "3) Click 'Remove' on that expert."
    ),
    "SELECT_HIRING_TEAM": (
        "PLAYBOOK: 1) Find the Hiring Team section/page. "
        "2) Look for member matching contains X with name equals Y. "
        "3) Click to view details."
    ),
    "CHOOSE_PROJECT_SIZE": (
        "PLAYBOOK: 1) Find the project size selector (in job posting or settings). "
        "2) Choose a size that is NOT the excluded one (e.g., NOT Small → choose Medium or Large)."
    ),
    "CLOSE_POST_A_JOB_WINDOW": (
        "PLAYBOOK: 1) Open the job posting form/window (click 'Post a Job'). "
        "2) Fill in rate_from >= X, rate_to <= Y. "
        "3) Make sure description does NOT contain Z. "
        "4) Close the window (X button or Cancel)."
    ),
    "NAVBAR_JOBS_CLICK": (
        "PLAYBOOK: 1) Find 'Jobs' link/tab in the navbar. 2) Click it."
    ),
    "NAVBAR_HIRES_CLICK": (
        "PLAYBOOK: 1) Find 'Hires' link/tab in the navbar. 2) Click it."
    ),
    "SEARCH_SKILL": (
        "PLAYBOOK: 1) Find the skill search bar. "
        "2) Type the query containing the given substring (e.g., '.N'). 3) Submit."
    ),
    "EDIT_PROFILE_LOCATION": (
        "PLAYBOOK: 1) Navigate to Profile/Settings. "
        "2) Find Location field. 3) Enter a location NOT containing the excluded string. 4) Save."
    ),
    "EDIT_PROFILE_EMAIL": (
        "PLAYBOOK: 1) Navigate to Profile/Settings/Account. "
        "2) Find Email field. 3) Enter an email NOT equal to the excluded one. 4) Save."
    ),
    # ---- AutoLodge (8007) ----
    "RESERVE_HOTEL": (
        "PLAYBOOK: 1) Browse hotel listings using list_cards. "
        "2) Find hotel matching ALL constraints (guests NOT X, location NOT contains Y, amenities NOT in list, title contains Z, rating <= W, reviews >= Q). "
        "3) Set guests count if needed. 4) Click 'Reserve' or 'Book Now'."
    ),
    "SEARCH_HOTEL": (
        "PLAYBOOK: 1) Find the hotel search bar. "
        "2) Type the search term CONTAINING the given substring (e.g., 'k, Ic'). 3) Submit."
    ),
    "PAYMENT_METHOD_SELECTED": (
        "PLAYBOOK: 1) Find the hotel matching constraints (ID NOT equals X, title NOT contains Y). "
        "2) Click to book/open payment. "
        "3) Select a payment method that does NOT contain 'card' (e.g., PayPal or Bank Transfer)."
    ),
    "EDIT_NUMBER_OF_GUESTS": (
        "PLAYBOOK: 1) Find the hotel/listing matching ALL constraints. "
        "2) Find the guest count selector (+/- buttons or dropdown). "
        "3) Set it to the required number (e.g., '2')."
    ),
    "SUBMIT_REVIEW": (
        "PLAYBOOK: 1) Find the listing matching constraints (host_name contains X, price < Y, amenities one of Z, name NOT contains W). "
        "2) Click 'Write Review' or 'Submit Review'. "
        "3) Set rating to exact value (e.g., 5.0). "
        "4) Type the review text EXACTLY as given. 5) Submit."
    ),
    "ADD_TO_WISHLIST_HOTEL": (
        "PLAYBOOK: 1) Find hotel matching constraints (rating equals X, guests < Y, amenities one of Z, price <= W). "
        "2) Click 'Add to Wishlist' or heart icon."
    ),
    "APPLY_FILTERS": (
        "PLAYBOOK: 1) Find filter controls (region, rating, price range). "
        "2) Set region/country to specified value. "
        "3) Set rating filter as specified. "
        "4) Apply the filter."
    ),
    "PEOPLE_DROPDOWN_OPENED": (
        "PLAYBOOK: 1) Find the people/guest selector on the main booking/search form. "
        "2) Click to open the dropdown. "
        "3) Select the number of people satisfying the constraint (e.g., >= 5 or equals 8)."
    ),
    "COUNTRY_SELECTED": (
        "PLAYBOOK: 1) Find the country/destination dropdown on the booking/search form. "
        "2) Make sure other filters (bookings, rating, people, date, time) are set per constraints. "
        "3) Open the country dropdown. 4) Select the specified country (e.g., 'Mexico' or 'South Africa')."
    ),
    # ---- AutoDelivery (8006) ----
    "RESTAURANT_NEXT_PAGE": (
        "PLAYBOOK: 1) Look for pagination at the bottom of the restaurant list. "
        "2) Click the 'Next' or '>' button."
    ),
    "RESTAURANT_PREV_PAGE": (
        "PLAYBOOK: 1) Look for pagination. 2) Click 'Previous' or '<' button."
    ),
    "SEARCH_DELIVERY_RESTAURANT": (
        "PLAYBOOK: 1) Find the restaurant search bar. "
        "2) For 'NOT contains X' → type any search term that does NOT include X. "
        "3) For 'contains X' → type exactly X as the query. 4) Submit."
    ),
    "DROPOFF_PREFERENCE": (
        "PLAYBOOK: 1) Find the order matching constraints (quantity > X, restaurant equals Y). "
        "2) Go to its cart/order page. "
        "3) Find the dropoff/delivery preference selector. "
        "4) Select an option NOT equal to the excluded one (e.g., NOT 'Text when arriving' → choose 'Hand it to me' or 'Leave at door')."
    ),
    "DELIVERY_PRIORITY_SELECTED": (
        "PLAYBOOK: 1) Find the order matching constraints (preferences contains X, quantity NOT Y, price equals Z, restaurant equals W). "
        "2) Find the delivery priority selector. "
        "3) Select a priority NOT equal to the excluded one."
    ),
    "VIEW_DELIVERY_RESTAURANT": (
        "PLAYBOOK: 1) Browse restaurant list. "
        "2) Find restaurant matching constraints (cuisine NOT X, description equals Y, rating < Z). "
        "3) Click on it to view details."
    ),
    "VIEW_ALL_RESTAURANTS": (
        "PLAYBOOK: 1) Click 'All Restaurants' or equivalent link/tab. "
        "2) The page should show all available restaurants."
    ),
    "OPEN_CHECKOUT_PAGE": (
        "PLAYBOOK: 1) Find order matching constraints (preferences NOT in list, size contains X, quantity NOT Y, price <= Z, restaurant contains W). "
        "2) Navigate to checkout for that order."
    ),
    # ---- AutoRestaurant (8003) ----
    "SEARCH_RESTAURANT": (
        "PLAYBOOK: 1) Find the restaurant search bar. "
        "2) Type the EXACT query from TASK_CONSTRAINTS (e.g., 'St. Lawrence' or 'Meadow Modern'). "
        "3) Submit search."
    ),
    "VIEW_RESTAURANT": (
        "PLAYBOOK: 1) Browse restaurant cards. "
        "2) Find restaurant matching constraints (cuisine NOT X, description equals Y, rating < Z). "
        "3) Click to view."
    ),
    "HELP_FAQ_TOGGLED": (
        "PLAYBOOK: 1) Navigate to Help/FAQ page. "
        "2) Find an FAQ item whose question does NOT contain the excluded text. "
        "3) Click to expand/toggle it."
    ),
    "HELP_VIEWED": (
        "PLAYBOOK: 1) Find the Help or FAQ link in navigation. 2) Click it to open."
    ),
    "ABOUT_FEATURE_CLICK": (
        "PLAYBOOK: 1) Navigate to the About page. "
        "2) Find the feature card containing the text (e.g., 'Trusted reviews'). "
        "3) Click on it."
    ),
    "CONTACT_FORM_SUBMIT": (
        "PLAYBOOK: 1) Navigate to Contact page. "
        "2) Fill: subject containing the required substring, plus other fields. "
        "3) Submit."
    ),
    # ---- AutoShop (8002) ----
    "CATEGORY_FILTER": (
        "PLAYBOOK: 1) Find the category filter sidebar or dropdown. "
        "2) Click the category that equals the specified value (e.g., 'technology'). "
        "3) Products should filter."
    ),
    "VIEW_WISHLIST": (
        "PLAYBOOK: 1) Find the Wishlist link/icon (often in nav or account menu). "
        "2) Click to view saved items."
    ),
    "PROCEED_TO_CHECKOUT": (
        "PLAYBOOK: 1) Go to cart. 2) Click 'Proceed to Checkout' button."
    ),
    "ORDER_COMPLETED": (
        "PLAYBOOK: 1) Find the item matching constraints (title contains X). "
        "2) Navigate to it. 3) Complete purchase/order process. Click 'Buy Now' or 'Add to Cart' → Checkout."
    ),
    "CAROUSEL_SCROLL": (
        "PLAYBOOK: 1) Find a carousel section whose title is NOT the excluded one. "
        "2) Click the left '<' scroll button on that carousel."
    ),
    "SHARE_PRODUCT": (
        "PLAYBOOK: 1) Find product matching constraints (brand contains X, price >= Y). "
        "2) Click 'Share' button on it."
    ),
    # ---- AutoDoc (8004) ----
    "ADD_CLIENT": (
        "PLAYBOOK: 1) Navigate to Clients section. "
        "2) Click 'Add New Client' button. "
        "3) Fill: name contains X, email NOT containing Y, matters < Z, status contains W, last NOT containing Q. "
        "4) Save."
    ),
    "ADD_NEW_MATTER": (
        "PLAYBOOK: 1) Navigate to Matters section. "
        "2) Click 'Add New Matter'. "
        "3) Fill: name NOT X, client NOT Y, status contains Z. "
        "4) Save."
    ),
    "SORT_MATTER_BY_CREATED_AT": (
        "PLAYBOOK: 1) Navigate to Matters list. "
        "2) Find the 'created_at' column header or sort button. "
        "3) Click it to sort in the direction specified (asc = earliest first)."
    ),
    "CHANGE_USER_NAME": (
        "PLAYBOOK: 1) Navigate to Settings or Profile. "
        "2) Find the user name/display name field. "
        "3) Set it to the specified value (where current name NOT contains X). "
        "4) Save."
    ),
    "VIEW_PENDING_EVENTS": (
        "PLAYBOOK: 1) Navigate to Calendar or Events section. "
        "2) Find pending/upcoming events. "
        "3) Find the one where earliest date NOT equals X. "
        "4) Click to view."
    ),
    # ---- General fallback ---
    "GENERAL": (
        "PLAYBOOK: Analyze the task carefully, identify the key action required, "
        "and execute the most direct path to complete it. "
        "Use TASK_CONSTRAINTS to find the correct item and fill forms. "
        "Use WEBSITE_CONTEXT for navigation hints."
    ),
    # ---- IWA-specific task types ---
    "ENTER_DESTINATION": (
        "PLAYBOOK: 1) Find the destination input field on the page (look for 'destination', 'to', 'where', "
        "or address-style input). "
        "2) Click it to focus. "
        "3) Clear the field if it has a pre-filled value. "
        "4) Type any valid destination address (e.g. '123 Main Street, New York, NY 10001') that is "
        "DIFFERENT from the forbidden value in TASK_CONSTRAINTS (the NOT constraint). "
        "5) Confirm or submit the destination."
    ),
    "EMAIL_MARK_SPAM": (
        "PLAYBOOK: 1) You are on an email/webmail interface. "
        "2) Use list_cards or search_text to find the email matching ALL constraints: "
        "check the subject (contains/not-contains) AND sender address (contains/not-contains). "
        "3) Click on that specific email to open it. "
        "4) Find the 'Spam', 'Mark as Spam', 'Report Spam', or 'Junk' button/option "
        "(often in a toolbar, kebab menu, or right-click context menu). "
        "5) Click it to mark the email as spam."
    ),
    "EMAIL_DELETE": (
        "PLAYBOOK: 1) Find the email matching all subject/sender constraints. "
        "2) Click it to select or open it. "
        "3) Find and click the Delete/Trash button."
    ),
    "EMAIL_OPEN": (
        "PLAYBOOK: 1) Find the email matching all constraints. "
        "2) Click on it to open and read it."
    ),
    "EMAIL_COMPOSE": (
        "PLAYBOOK: 1) Click Compose/New Message/Reply/Forward. "
        "2) Fill in To, Subject, Body fields with EXACT values from task. "
        "3) Send."
    ),
    "DELETE_TASK": (
        "PLAYBOOK: 1) Navigate to the task/todo list page. "
        "2) Use list_cards tool to see all tasks with their fields (name, description, date, priority). "
        "3) Find the task matching ALL TASK_CONSTRAINTS: "
        "   - name NOT contains the excluded string "
        "   - description contains the required substring "
        "   - date less than the given date "
        "   - priority equals the given value. "
        "4) Click that task's Delete/Remove/Trash button. "
        "5) Confirm deletion if a dialog appears."
    ),
    "CREATE_TASK": (
        "PLAYBOOK: 1) Find the 'New Task', 'Add Task', or '+' button. "
        "2) Fill in name, description, date, priority fields with EXACT values from task. "
        "3) Save/Submit."
    ),
    "EDIT_TASK": (
        "PLAYBOOK: 1) Find the task matching the constraints. "
        "2) Click Edit/Pencil icon. "
        "3) Update the specified fields with EXACT values. "
        "4) Save."
    ),
    "COMPLETE_TASK": (
        "PLAYBOOK: 1) Find the task matching the constraints. "
        "2) Click the Complete/Done/Checkmark button on it."
    ),
    "BOOKING_CONFIRM": (
        "PLAYBOOK: 1) You are on a lodging/accommodation booking site. "
        "2) Use list_cards to browse listings. Find the one matching ALL TASK_CONSTRAINTS: "
        "   - rating equals (exact), price greater_than, location contains, "
        "   - host_name contains, reviews equals, amenities NOT in list, "
        "   - title NOT equals, guests_set equals. "
        "3) Set guests count to the required value (guests_set constraint). "
        "4) Click 'Book Now', 'Reserve', or 'Confirm' on that listing. "
        "5) Fill the payment form: "
        "   - For fields with 'equals' constraint: type EXACTLY that value "
        "   - For fields with 'not equals' constraint: enter any valid alternative "
        "     (e.g. card_number not_equals X -> use '4111111111111111'; "
        "      expiration not_equals Y -> use '12/27'). "
        "   - cvv, zipcode, country: use EXACT values from TASK_CONSTRAINTS. "
        "6) Submit/Confirm the booking."
    ),
    "JOB_POSTING": (
        "PLAYBOOK: 1) Look for a 'Post a Job', 'Create Job', 'Add Job', or '+' button/link. "
        "2) Click it to open the job posting form/page. "
        "3) Find the job title field. "
        "4) Type the EXACT job title from TASK_CREDENTIALS['job_title'] or the task text. "
        "5) Submit/post the job if required."
    ),
    "WRITE_JOB_TITLE": (
        "PLAYBOOK: 1) Look for 'Post a Job' or '+' button to start a job posting. "
        "2) Click it to open the job title input. "
        "3) If TASK_CREDENTIALS has 'job_title', type that EXACTLY. "
        "4) If TASK_CREDENTIALS has 'job_title_contains', type a job title that CONTAINS that substring "
        "   (e.g. if contains='opers J' type 'Developers Jobs'). "
        "5) Do NOT click submit/publish - just fill the title field as instructed."
    ),
    # ---- AutoRide (8012) ----
    "ENTER_LOCATION": (
        "PLAYBOOK: 1) Find the location/pickup input field (labeled 'location', 'from', 'pickup'). "
        "2) Click it to focus. "
        "3) Type the EXACT location string from TASK_CONSTRAINTS (equals constraint). "
        "4) Wait for/click the matching autocomplete suggestion that matches exactly. "
        "5) Confirm selection."
    ),
    "SEARCH_RIDE": (
        "PLAYBOOK: 1) You are on AutoRide. Find the ride search/filter interface. "
        "2) Constraints specify location, destination, and scheduled_time. "
        "3) Apply filters or scroll through rides to find one matching ALL constraints. "
        "4) For 'location NOT contains X': choose a location that does NOT include the excluded text. "
        "5) For 'destination contains Y': find a ride whose destination matches. "
        "6) For 'scheduled_time less_equal Z': pick a time at or before the given datetime. "
        "7) Click on that matching ride entry to view details."
    ),
    "SELECT_TIME": (
        "PLAYBOOK: 1) Find the time picker/selector on the trip booking page. "
        "2) Look at the constraint: if 'GREATER THAN OR EQUAL TO X', select a time >= X. "
        "   If 'LESS THAN X', select a time < X. "
        "3) Click/select the appropriate time slot that satisfies the constraint."
    ),
    # ---- AutoCalendar (8010) ----
    "SELECT_DAY": (
        "PLAYBOOK: 1) Find the calendar view buttons (Day, Week, Month, 5-day). "
        "2) Click the 'Day' button to switch to day view."
    ),
    "EVENT_WIZARD_OPEN": (
        "PLAYBOOK: 1) Find the 'Add Event', '+ New Event', or '+' button on the calendar page. "
        "2) Click it to open the event creation wizard/dialog. "
        "3) If a title field appears, type a title that satisfies the CONTAINS constraint from the task. "
        "4) Do NOT necessarily submit - the task may just require opening the wizard."
    ),
    "CELL_CLICKED": (
        "PLAYBOOK: 1) Switch to the '5 days' view if not already there. "
        "2) The task specifies a date AFTER X and a time BEFORE Y. "
        "3) Find a cell in the calendar grid that is on a date after the given date "
        "   AND in a time slot before the given time. "
        "4) Click on that cell to select/create an event at that time."
    ),
    "EVENT_REMOVE_ATTENDEE": (
        "PLAYBOOK: 1) Find an existing event on the calendar. "
        "2) Click it to open event details. "
        "3) Find the attendees list. "
        "4) Find the attendee whose email does NOT match the excluded email. "
        "5) Click Remove/Delete next to that attendee."
    ),
    "SEARCH_SUBMIT": (
        "PLAYBOOK: 1) Find the search input on the calendar/current page. "
        "2) Type the search query - use the value from TASK_CONSTRAINTS (contains constraint). "
        "3) Press Enter or click the search button to submit."
    ),
    # ---- AutoShop (8002) ----
    "VIEW_DETAIL": (
        "PLAYBOOK: 1) Browse the product listing on AutoShop. "
        "2) Use TASK_CONSTRAINTS to find the right product: brand equals/contains AND price constraints. "
        "3) Click on the product card to open its detail page. "
        "4) The task is complete when the product detail page is shown."
    ),
    "QUANTITY_CHANGED": (
        "PLAYBOOK: 1) Navigate to the shopping cart (cart icon, top-right). "
        "2) Find the item matching the title constraint (title equals X or NOT 'Y'). "
        "3) Find the quantity input/stepper for that item. "
        "4) Change the quantity to the specified new_quantity value. "
        "5) Confirm/update the cart."
    ),
    "ITEM_INCREMENTED": (
        "PLAYBOOK: 1) Navigate to the cart or the item's cart section. "
        "2) Find the item quantity control (+/- buttons or number input). "
        "3) Increment the quantity to reach the target value (e.g. 5). "
        "4) Confirm the quantity update."
    ),
    # ---- AutoConnect (8008) ----
    "SAVE_POST": (
        "PLAYBOOK: 1) On AutoConnect, browse the feed/posts. "
        "2) Find the post matching ALL constraints: author CONTAINS X AND content CONTAINS Y. "
        "3) Find the Save/Bookmark button on that post (often a bookmark icon). "
        "4) Click it to save the post."
    ),
    "HOME_NAVBAR": (
        "PLAYBOOK: 1) Find the navigation bar at the top. "
        "2) Look for 'Home' tab/link. "
        "3) Click it to navigate to the Home section."
    ),
    "VIEW_HIDDEN_POSTS": (
        "PLAYBOOK: 1) Go to your profile or account settings on AutoConnect. "
        "2) Look for 'Hidden Posts', 'Privacy', or 'Activity' section. "
        "3) Navigate to the hidden posts section to view them."
    ),
    "SEARCH_JOBS": (
        "PLAYBOOK: 1) On AutoConnect, find the Jobs section (via navbar or search). "
        "2) Find the search/filter input for jobs. "
        "3) Type a search query that does NOT contain the excluded term (from NOT CONTAIN constraint). "
        "4) Submit the search."
    ),
    "APPLY_FOR_JOB": (
        "PLAYBOOK: 1) On AutoConnect (jobs site), browse the job listings. "
        "2) Find a job matching ALL TASK_CONSTRAINTS: "
        "   - job_title NOT CONTAIN excluded term, company NOT CONTAIN excluded term, etc. "
        "3) Click on that job listing to open its detail page. "
        "4) Find and click the 'Apply' button to apply for the job."
    ),
    # ---- AutoMail (8005) ----
    "MARK_AS_SPAM": (
        "PLAYBOOK: 1) On AutoMail, browse the inbox/email list. "
        "2) Find the email matching ALL constraints: "
        "   - subject CONTAINS the required substring "
        "   - from_email does NOT CONTAIN the excluded substring. "
        "3) Click on that email to open it, OR select it with a checkbox. "
        "4) Find 'Mark as Spam', 'Report Spam', or 'Spam' button (toolbar or menu). "
        "5) Click it."
    ),
    "MARK_AS_UNREAD": (
        "PLAYBOOK: 1) Find the email matching ALL constraints: "
        "   - from_email equals the given address "
        "   - is_read equals False (already unread? - still find it). "
        "2) Open the email or right-click/use menu. "
        "3) Click 'Mark as Unread' option."
    ),
    "VIEW_EMAIL": (
        "PLAYBOOK: 1) Browse the email list. "
        "2) Find the email matching the constraint (subject NOT equals X, or subject contains Y). "
        "3) Click on it to open and view it."
    ),
    "THEME_CHANGED": (
        "PLAYBOOK: 1) On AutoMail, find the Settings/Preferences option (gear icon or settings menu). "
        "2) Look for 'Theme', 'Appearance', or 'Display' settings. "
        "3) Select 'Dark' theme. "
        "4) Save/Apply the setting."
    ),
    # ---- AutoLodge (8007) ----
    "SHARE_HOTEL": (
        "PLAYBOOK: 1) Browse hotel listings on AutoLodge. "
        "2) Find the hotel matching ALL constraints: location, host_name NOT contains, price > X, "
        "   title NOT contains, amenities include Y, guests > Z. "
        "3) Click on that hotel card. "
        "4) Find the Share button on the hotel detail page. "
        "5) Enter the recipient email address and send/share."
    ),
    "POPULAR_HOTELS_VIEWED": (
        "PLAYBOOK: 1) On AutoLodge, look for a 'Popular Hotels' or 'Featured' section. "
        "2) Apply filter for rating >= the specified value if a filter UI exists. "
        "3) Click to view popular hotels or apply the rating filter."
    ),
    "EDIT_NUMBER_OF_GUESTS": (
        "PLAYBOOK: 1) Browse hotel listings. "
        "2) Find the listing matching ALL TASK_CONSTRAINTS: "
        "   - guests_to less_than X "
        "   - rating greater_than Y "
        "   - title contains required substrings "
        "   - amenities NOT in the excluded list "
        "   - location NOT equals Z. "
        "3) Click on that listing to open it. "
        "4) Find the guests/people selector. "
        "5) Set guests count to the specified target number (e.g. 2). "
        "6) Confirm the change."
    ),
    # ---- AutoDelivery (8006) ----
    "REVIEW_SUBMITTED": (
        "PLAYBOOK: 1) On AutoDelivery, find the restaurant matching constraints: "
        "   name contains X, OR the review fields. "
        "2) Open the restaurant page or a past order. "
        "3) Find the 'Write Review' or 'Rate' button. "
        "4) Fill: author='James' (or as specified), rating=5 (or as specified, meeting >= constraint). "
        "5) Submit the review."
    ),
    "BACK_TO_ALL_RESTAURANTS": (
        "PLAYBOOK: 1) You are viewing a restaurant detail page on AutoDelivery. "
        "2) The task specifies which restaurant: find one matching name equals X and cuisine contains Y. "
        "3) Navigate to that restaurant's detail page. "
        "4) Find the 'Back', '< All Restaurants', or 'Back to Restaurants' button. "
        "5) Click it to return to the restaurant list."
    ),
    "ADDRESS_ADDED": (
        "PLAYBOOK: 1) On AutoDelivery, find the delivery address section or settings. "
        "2) Click 'Add Address' or similar. "
        "3) Type the exact address string from the task: e.g. '202 Birch Lane, Lakeview'. "
        "4) Fill any additional fields: preferences (contains required text), quantity, etc. "
        "5) Save/confirm the address."
    ),
    # ---- AutoRestaurant (8003) ----
    "COLLAPSE_MENU": (
        "PLAYBOOK: 1) Browse restaurants on AutoRestaurant. "
        "2) Find the restaurant matching constraints (rating NOT X, bookings equals Y, name NOT Z). "
        "3) Click on that restaurant to view its menu. "
        "4) Find the expanded menu section with a collapse/hide toggle. "
        "5) Click it to collapse the menu."
    ),
    "CONTACT_CARD_CLICK": (
        "PLAYBOOK: 1) Find the contact cards/methods section on the page. "
        "2) Find the card whose type does NOT contain the excluded value (e.g. NOT 'Email'). "
        "3) Click on that contact card."
    ),
    "SCROLL_VIEW": (
        "PLAYBOOK: 1) Find the scrollable section that does NOT contain the excluded section name. "
        "2) Scroll in the specified direction ('left' or 'right') on that section/carousel."
    ),
    "HELP_PAGE_VIEW": (
        "PLAYBOOK: 1) Find the Help/FAQ link in the navigation or footer. "
        "2) Click it to navigate to the Help page."
    ),
    "HELP_CATEGORY_SELECTED": (
        "PLAYBOOK: 1) Navigate to the Help page. "
        "2) Find the category matching the equals constraint (e.g. 'Reservations'). "
        "3) Click on that help category."
    ),
    # ---- AutoDoc (8004) ----
    "SEARCH_MATTER": (
        "PLAYBOOK: 1) On AutoDoc, find the Matters search bar/input. "
        "2) Type a search query that does NOT contain the excluded term. "
        "3) Submit the search to find matching matters."
    ),
    "FILTER_CLIENTS": (
        "PLAYBOOK: 1) On AutoDoc Clients page, find filter/search options. "
        "2) Apply filters: status NOT 'X', matters NOT equals 'Y'. "
        "3) Show the filtered client list."
    ),
    "FILTER_MATTER_STATUS": (
        "PLAYBOOK: 1) On AutoDoc Matters page, find the status filter dropdown or search. "
        "2) Filter by status that contains the required substring (e.g. 'ived' → 'Archived'). "
        "3) Apply/submit the filter."
    ),
    "DOCUMENT_DELETED": (
        "PLAYBOOK: 1) Navigate to Documents section on AutoDoc. "
        "2) Find the document matching constraints: status NOT CONTAINS X, size > Y. "
        "3) Select/click that document. "
        "4) Find the Delete/Remove button and click it. "
        "5) Confirm deletion."
    ),
    # ---- AutoMedic (8013) ----
    "DOCTOR_CONTACTED_SUCCESSFULLY": (
        "PLAYBOOK: 1) On AutoMedic, find the doctor matching ALL constraints: "
        "   doctor_name, patient_name, subject NOT equals, urgency NOT equals, "
        "   preferred_contact_method contains, patient_phone, patient_email NOT equals. "
        "2) Open the Contact Doctor form for that doctor. "
        "3) Fill in: patient_name (exact), patient_phone (exact), patient_email (NOT the excluded email, use any valid one). "
        "4) Subject: NOT the excluded subject - use any valid medical subject. "
        "5) Urgency: NOT the excluded value - pick any other option. "
        "6) Preferred contact: must CONTAIN the specified substring (e.g. 'ither' → 'Either'). "
        "7) Submit the form."
    ),
    "VIEW_DOCTOR_AVAILABILITY": (
        "PLAYBOOK: 1) On AutoMedic, browse doctors list. "
        "2) Find the doctor matching ALL constraints: "
        "   doctor_name NOT X, speciality CONTAINS Y, rating NOT Z, "
        "   consultation_fee <= W, language equals V. "
        "3) Click on that doctor's card/profile. "
        "4) Navigate to the Availability tab/section to view their schedule."
    ),
    # ---- AutoList (8011) ----
    "AUTOLIST_SELECT_TASK_PRIORITY": (
        "PLAYBOOK: 1) On AutoList, find the task list. "
        "2) Find a task whose current priority is NOT the excluded value. "
        "3) Click on that task's priority selector/dropdown. "
        "4) Select 'High' (or the target priority value). "
        "5) Save/confirm."
    ),
    "AUTOLIST_CANCEL_TASK_CREATION": (
        "PLAYBOOK: 1) Start creating a new task: click 'Add Task' or '+'. "
        "2) Fill in the fields as specified (name equals X, description NOT contains Y, date equals Z, priority equals W). "
        "3) Instead of submitting, find and click 'Cancel', 'Discard', or 'X' to cancel the creation. "
        "4) Do NOT save/submit the task."
    ),
    "AUTOLIST_TEAM_CREATED": (
        "PLAYBOOK: 1) Navigate to the Teams section on AutoList. "
        "2) Click 'Create Team' or '+'. "
        "3) Fill in: name (must CONTAIN required substring), description (exact value from task), "
        "   add member whose name contains required substring, assign role that is NOT the excluded value. "
        "4) Save/create the team."
    ),
    # ---- AutoBooks (8001) ----
    "DELETE_BOOK": (
        "PLAYBOOK: 1) Login with the placeholder credentials (username=' ', password=' '). "
        "2) Navigate to your books or the book matching the id constraint. "
        "3) Find the book matching id equals X (from task or credentials). "
        "4) Click Delete/Remove button. "
        "5) Confirm deletion."
    ),
    "EDIT_BOOK": (
        "PLAYBOOK: 1) Login with the credentials from the task (username, password). "
        "2) Find the book matching ALL constraints: page_count >= X, author contains Y, rating <= Z, genres equals W. "
        "3) Click Edit/Pencil icon on that book. "
        "4) Modify the required fields. "
        "5) Save changes."
    ),
    "REMOVE_FROM_READING_LIST": (
        "PLAYBOOK: 1) Login with the placeholder credentials. "
        "2) Navigate to your Reading List. "
        "3) Find a book whose name is NOT the excluded title AND description is NOT the excluded description. "
        "4) Click Remove/Delete from reading list on that book."
    ),
    "CONTACT_BOOK": (
        "PLAYBOOK: 1) Navigate to the Contact page on AutoBooks. "
        "2) Fill in the form: email NOT 'user1@site.com' (use any other email), "
        "   subject NOT contains 'Order', name NOT contains 'Alice', "
        "   message contains the required text (e.g. 'I\\'m writing to request support'). "
        "3) Submit the contact form."
    ),
    "REGISTRATION_BOOK": (
        "PLAYBOOK: 1) Navigate to the Register page on AutoBooks. "
        "2) Fill in username, email, password with the placeholder values (often empty or space ' '). "
        "3) Submit to register."
    ),
    # ---- AutoCinema (8000) ----
    "ADD_TO_WATCHLIST": (
        "PLAYBOOK: 1) Login first if required (username='user ', password='Passw0rd!'). "
        "2) Browse the films list. "
        "3) Find the film matching the constraint (genres NOT CONTAINS X means choose a film without that genre). "
        "4) Click on that film to open its detail page. "
        "5) Click the 'Add to Watchlist' button."
    ),
    "SHARE_MOVIE": (
        "PLAYBOOK: 1) Browse films on AutoCinema. "
        "2) Find the film matching the constraint (duration equals X minutes). "
        "3) Click on that film to open its detail page. "
        "4) Find and click the Share button. "
        "5) Complete the share action."
    ),
    # ---- AutoHire (8009) ----
    "EDIT_ABOUT": (
        "PLAYBOOK: 1) Navigate to your profile on AutoHire. "
        "2) Find the 'About' or 'Bio' section. "
        "3) Click Edit/pencil icon to enter edit mode. "
        "4) If constraint is NOT contains X: clear the field and type any text that does NOT contain X. "
        "5) If constraint is description equals X: type exactly that description. "
        "6) Save the changes."
    ),
    # ---- AutoBooks (8001) additional ----
    "BOOK_DETAIL": (
        "PLAYBOOK: 1) On AutoBooks, browse the books list. "
        "2) Find a book matching ALL TASK_CONSTRAINTS: "
        "   rating NOT 'X' (pick a different rating), genres NOT CONTAIN 'Y', page_count <= Z. "
        "3) Click on that book to open its detail page. "
        "4) Ensure the detail/info page is fully visible."
    ),
    "FILTER_BOOK": (
        "PLAYBOOK: 1) On AutoBooks, find the filter/genre dropdown or filter panel. "
        "2) Select the genre specified: genres equals 'Dystopian' means select 'Dystopian'. "
        "3) Apply the filter. "
        "4) Verify filtered results appear."
    ),
    "SEARCH_BOOK": (
        "PLAYBOOK: 1) On AutoBooks, find the search bar. "
        "2) Type the exact query from TASK_CONSTRAINTS (e.g. 'Harry Potter and the Chamber of Secrets'). "
        "3) Press Enter or click Search. "
        "4) Verify search results appear."
    ),
    "LOGIN_BOOK": (
        "PLAYBOOK: 1) On AutoBooks, click Login. "
        "2) If credentials say '<username>' or '<password>', use 'user' and 'Passw0rd!' as fallback. "
        "3) Type username and password into fields. "
        "4) Click Login/Sign In button."
    ),
    "LOGOUT_BOOK": (
        "PLAYBOOK: 1) On AutoBooks, first login: if username is '<username>' use 'user', "
        "   if password is '<password>' use 'Passw0rd!'. "
        "2) After login completes, find the logout/sign-out option. "
        "3) Click Logout."
    ),
    "VIEW_CART_BOOK": (
        "PLAYBOOK: 1) On AutoBooks, if username/password are provided (even as '<username>'), login first. "
        "   Use 'user' for '<username>' and 'Passw0rd!' for '<password>' as fallback. "
        "2) After login, click the Cart icon or navigate to /cart. "
        "3) View the cart contents."
    ),
    # ---- AutoShop (8002) additional ----
    "CHECKOUT_STARTED": (
        "PLAYBOOK: 1) On AutoShop, browse the products. "
        "2) Find a product where total_amount satisfies the constraint (less_equal X, greater_equal Y). "
        "3) Click 'Buy Now' button on that product. "
        "4) This should initiate checkout - confirm checkout page appears."
    ),
    # ---- AutoRestaurant (8003) additional ----
    "ABOUT_PAGE_VIEW": (
        "PLAYBOOK: 1) On AutoRestaurant, find the 'About' link in navbar or footer. "
        "2) Click 'About' to navigate to the about page. "
        "3) Verify the About page content is visible."
    ),
    "DATE_DROPDOWN_OPENED": (
        "PLAYBOOK: 1) On AutoRestaurant, find the date/time reservation selector. "
        "2) Click on the date selector to open the dropdown. "
        "3) Select a date satisfying the constraint (less_equal given date). "
        "4) Confirm selection."
    ),
    "CONTACT_FORM_SUBMIT": (
        "PLAYBOOK: 1) On AutoRestaurant, navigate to the Contact page. "
        "2) Fill in: email CONTAINS 'olivia.brown@' (or exact), username NOT CONTAINS 'Olivia'. "
        "3) Fill any remaining fields. "
        "4) Submit the contact form."
    ),
    # ---- AutoDoc (8004) additional ----
    "LOG_EDITED": (
        "PLAYBOOK: 1) On AutoDoc, navigate to Logs/Time entries or the Matter page. "
        "2) Find the log entry where matter CONTAINS the given substring (e.g. 'Merger'). "
        "3) Click Edit/pencil icon on that log. "
        "4) Make any change (or just click save) to mark it edited."
    ),
    "ARCHIVE_MATTER": (
        "PLAYBOOK: 1) On AutoDoc, navigate to Matters list. "
        "2) Find a matter where status NOT CONTAINS the excluded value (e.g. 'On Hold'). "
        "3) Click on that matter to select it. "
        "4) Find and click 'Archive' button/option. "
        "5) Confirm archiving if prompted."
    ),
    "VIEW_CLIENT_DETAILS": (
        "PLAYBOOK: 1) On AutoDoc, navigate to Clients. "
        "2) Find client matching ALL constraints: email equals X, matters equals Y, status contains Z. "
        "3) Click on that client to open their detail page. "
        "4) Ensure client details are fully visible."
    ),
    "VIEW_MATTER_DETAILS": (
        "PLAYBOOK: 1) On AutoDoc, navigate to Matters. "
        "2) Find matter matching ALL constraints: name equals X, status contains Y. "
        "3) Click on that matter to open its detail page. "
        "4) Ensure matter details are visible."
    ),
    # ---- AutoMail (8005) additional ----
    "SEND_EMAIL": (
        "PLAYBOOK: 1) On AutoMail, click Compose/New Email. "
        "2) In the To/recipient field, type the address from the task (e.g. 'recipient@example.com'). "
        "3) In the Subject field, type a subject that CONTAINS the required substring. "
        "4) In the Body, type text that CONTAINS the required substring. "
        "5) Click Send."
    ),
    "SEARCH_EMAIL": (
        "PLAYBOOK: 1) On AutoMail, find the Search bar. "
        "2) Type the query from TASK_CONSTRAINTS (e.g. query equals '2'). "
        "3) Press Enter or click Search. "
        "4) Verify search results appear."
    ),
    # ---- AutoDelivery (8006) additional ----
    "RESTAURANT_FILTER": (
        "PLAYBOOK: 1) On AutoDelivery, find the cuisine filter/dropdown. "
        "2) For NOT CONTAIN 'Portuguese': do NOT select Portuguese - select any other cuisine or use the filter. "
        "3) Apply the filter and verify filtered restaurants appear."
    ),
    "ADD_TO_CART_MENU_ITEM": (
        "PLAYBOOK: 1) On AutoDelivery, browse restaurants. "
        "2) Find a restaurant that CONTAINS the required name. "
        "3) Find a menu item matching constraints: preferences in list, size equals X, quantity <= Y. "
        "4) Add that item to cart."
    ),
    "ADD_TO_CART_MODAL_OPEN": (
        "PLAYBOOK: 1) On AutoDelivery, find the restaurant and menu item matching constraints. "
        "   price <= X, item equals 'Chef's Special', restaurant equals 'Candlenut'. "
        "2) Click on that item to open the add-to-cart modal. "
        "3) The modal should be visible - set quantity/size if needed."
    ),
    "QUICK_ORDER_STARTED": (
        "PLAYBOOK: 1) On AutoDelivery, look for a 'Quick Order' button on any restaurant card. "
        "2) Click Quick Order on any restaurant. "
        "3) Verify the quick order flow starts."
    ),
    # ---- AutoLodge (8007) additional ----
    "MESSAGE_HOST": (
        "PLAYBOOK: 1) On AutoLodge, use list_cards to find a hotel matching ALL constraints: "
        "   host_name NOT 'Kevin', price < 601, guests < 3, amenities in ['Balcony views'], "
        "   rating < 5.69, location equals 'Madrid, Spain', title NOT 'Mount Nelson Hotel'. "
        "2) Click on that listing to open it. "
        "3) Find 'Message Host' button. "
        "4) Type a message that CONTAINS the required text (e.g. 'm'). "
        "5) Send the message."
    ),
    "EDIT_CHECK_IN_OUT_DATES": (
        "PLAYBOOK: 1) On AutoLodge, find the listing matching ALL constraints: "
        "   checkin date <= given, checkout date <= given, guests_set NOT equals X, "
        "   amenities contains Y, price < Z, host_name equals W. "
        "2) Open that listing's booking/reservation form. "
        "3) Modify the check-in and check-out dates to match constraints. "
        "4) Save/confirm."
    ),
    "WISHLIST_OPENED": (
        "PLAYBOOK: 1) On AutoLodge, find the Wishlist/Saved Hotels icon or menu item. "
        "2) Click it to open your wishlist. "
        "3) Verify saved/wishlisted hotels are visible."
    ),
    # ---- AutoConnect (8008) additional ----
    "EDIT_PROFILE": (
        "PLAYBOOK: 1) On AutoConnect, navigate to your Profile (click avatar or 'My Profile'). "
        "2) Click Edit Profile / pencil icon. "
        "3) Find the bio/about field. "
        "4) If constraint is bio NOT CONTAIN 'X': clear the bio and type any text that does NOT contain X. "
        "5) Save changes."
    ),
    "POST_STATUS": (
        "PLAYBOOK: 1) On AutoConnect, find the status/post input area on the home/feed page. "
        "2) Click in the text box. "
        "3) Type content that CONTAINS the required text (e.g. 'i'). "
        "4) Click Post/Submit."
    ),
    # ---- AutoHire (8009) additional ----
    "NAVBAR_EXPERTS_CLICK": (
        "PLAYBOOK: 1) On AutoHire, look at the top navigation bar. "
        "2) Find and click the 'Experts' link/option. "
        "3) Verify the experts list page loads."
    ),
    "ADD_SKILL": (
        "PLAYBOOK: 1) On AutoHire, navigate to your Profile or Skills section. "
        "2) Find 'Add Skill' button. "
        "3) Type a skill name that does NOT contain the excluded substring. "
        "4) Save/confirm."
    ),
    "SUBMIT_JOB": (
        "PLAYBOOK: 1) On AutoHire, navigate to 'Post a Job' or 'Submit Job'. "
        "2) Fill in: title NOT equal to 'DevOps Jobs' (use a different title), "
        "   rate_from LESS than 29, rate_to GREATER EQUAL 49. "
        "3) Fill other required fields. "
        "4) Submit the job posting."
    ),
    "HIRE_LATER_START": (
        "PLAYBOOK: 1) On AutoHire, navigate to 'Hire Later' page (saved/deferred experts). "
        "2) Find the expert matching ALL constraints: "
        "   country contains X, role contains Y, name equals Z (or NOT contains W). "
        "3) Click 'Start Hiring' or 'Hire Now' button for that expert."
    ),
    # ---- AutoCalendar (8010) additional ----
    "SELECT_CALENDAR": (
        "PLAYBOOK: 1) On AutoCalendar, find the calendar list/sidebar. "
        "2) Find the calendar whose name CONTAINS the required substring (e.g. 'c'). "
        "3) Click on it / check its checkbox to select it. "
        "4) Verify it is selected (checkbox checked, events visible)."
    ),
    "UNSELECT_CALENDAR": (
        "PLAYBOOK: 1) On AutoCalendar, find the calendar list/sidebar. "
        "2) Find the calendar whose name CONTAINS the required substring (e.g. 'uran'). "
        "3) Click on it / uncheck its checkbox to deselect/unselect it."
    ),
    # ---- AutoList (8011) additional ----
    "AUTOLIST_COMPLETE_TASK": (
        "PLAYBOOK: 1) On AutoList, find the Tasks section. "
        "2) Find the task matching ALL constraints: "
        "   name equals 'Deploy to staging', description NOT contains 'phk', "
        "   date >= '2026-03-08', priority NOT equals 'Highest'. "
        "3) Click the 'Complete' / checkmark button on that task. "
        "4) Confirm completion."
    ),
    # ---- AutoRide (8012) additional ----
    "TRIP_DETAILS": (
        "PLAYBOOK: 1) On AutoRide, view your trips/rides list. "
        "2) Find the trip matching ALL constraints: "
        "   location CONTAINS/NOT CONTAINS X, destination CONTAINS/NOT CONTAINS Y, "
        "   ride_name CONTAINS/EQUALS Z, scheduled_time GREATER/NOT EQUAL to W. "
        "3) Click on that trip to view its details."
    ),
    "SELECT_CAR": (
        "PLAYBOOK: 1) On AutoRide, find the ride matching ALL constraints: "
        "   location NOT CONTAIN X, destination CONTAINS Y, "
        "   ride_name NOT EQUAL Z, scheduled_time NOT EQUAL W. "
        "2) Click on that ride to open it. "
        "3) Select the car/vehicle option."
    ),
    "SEARCH_DESTINATION": (
        "PLAYBOOK: 1) On AutoRide, find the destination search bar. "
        "2) Type ANY destination that is NOT the excluded value from the constraint. "
        "3) Press Enter or click Search."
    ),
    "SELECT_DATE": (
        "PLAYBOOK: 1) On AutoRide, find the date picker for your trip. "
        "2) Select the exact date from TASK_CONSTRAINTS (e.g. '2026-03-23'). "
        "3) Confirm the date selection."
    ),
    # ---- AutoMedic (8013) additional ----
    "REFILL_PRESCRIPTION": (
        "PLAYBOOK: 1) On AutoMedic, navigate to Prescriptions section. "
        "2) Find a prescription matching ALL constraints: "
        "   medicine_name NOT contains X, doctor_name NOT contains Y. "
        "3) Click 'Refill' button on that prescription. "
        "4) Confirm the refill action."
    ),
    "VIEW_PRESCRIPTION": (
        "PLAYBOOK: 1) On AutoMedic, navigate to Prescriptions. "
        "2) Find prescription matching ALL constraints: "
        "   medicine_name equals X, doctor_name NOT contains Y, start_date NOT equals Z, "
        "   category NOT contains W, status NOT equals V, dosage contains U. "
        "3) Click on that prescription to view its details."
    ),
    "FILTER_DOCTOR_REVIEWS": (
        "PLAYBOOK: 1) On AutoMedic, navigate to the Reviews or Doctors section. "
        "2) Find the filter for doctor reviews. "
        "3) Set: doctor_name NOT CONTAINS X, filter_rating EQUALS Y (e.g. '4.0'), "
        "   sort_order NOT CONTAINS Z, speciality CONTAINS W. "
        "4) Apply the filter."
    ),
    # --- General fallback ---
    "GENERAL": (
        "PLAYBOOK: Analyze the task carefully, identify the key action required, "
        "and execute the most direct path to complete it. "
        "Use TASK_CONSTRAINTS to find the correct item and fill forms."
    ),
}


def _detect_website(url: str) -> str:
    """Map URL port to website name for the 14 IWA websites."""
    m = re.search(r":(\d+)/", url or "")
    if m:
        port = int(m.group(1))
        return {
            8000: "AutoCinema",
            8001: "AutoBooks",
            8002: "AutoShop",
            8003: "AutoRestaurant",
            8004: "AutoDoc",
            8005: "AutoMail",
            8006: "AutoDelivery",
            8007: "AutoLodge",
            8008: "AutoConnect",
            8009: "AutoHire",
            8010: "AutoCalendar",
            8011: "AutoList",
            8012: "AutoRide",
            8013: "AutoMedic",
        }.get(port, f"Unknown:{port}")
    return "Unknown"


def _website_context(website: str) -> str:
    """Return UI structure / navigation hints for each IWA website."""
    ctx: Dict[str, str] = {
        "AutoCinema": (
            "SITE: Movie/film database. NAV: Films list, Login/Register, Admin panel (when logged in). "
            "Film cards show title, year, genre, director, duration. "
            "Click film → detail page with Watch Trailer button, Add to Watchlist button, Share button, Comments section. "
            "Admin: Add Film, Edit Film, Delete Film (requires login with 'user '/'Passw0rd!'). "
            "Registration: username='newuser ', email='newuser @gmail.com', password='Passw0rd!'."
        ),
        "AutoBooks": (
            "SITE: Book store. NAV: Books, Cart icon, Login/Register. "
            "Books have title, author, genres, year, page_count, rating, price. "
            "Login/Register with placeholder credentials (' '/' '). "
            "Book detail: Add to Cart, Add to Wishlist, Open Preview buttons. "
            "Admin: Add Book, Edit Book, Delete Book. Cart icon top-right."
        ),
        "AutoShop": (
            "SITE: E-commerce store. NAV: Products grid, Category sidebar/filter, Cart icon, Wishlist. "
            "Products have name, brand, price, description, rating. "
            "Category filter on left sidebar (click to filter by category). "
            "Product card: Add to Cart, Add to Wishlist, Share buttons. "
            "Cart page: shows items, total, Proceed to Checkout button. "
            "Carousel sections: scroll left/right buttons on carousel cards."
        ),
        "AutoRestaurant": (
            "SITE: Restaurant reservation/booking. NAV: Restaurants, About, Help/FAQ, Contact. "
            "Main page: search bar, country selector dropdown, date/time pickers, people/guest count. "
            "Guest dropdown: click on people/guest count to open dropdown and select number. "
            "Restaurant cards: click to view details. "
            "Help/FAQ page: expandable FAQ items. "
            "About page: feature cards (Trusted reviews, etc.). "
            "Contact form: name, email, message, subject fields."
        ),
        "AutoDoc": (
            "SITE: Legal case management + calendar. NAV: Dashboard, Matters, Clients, Calendar. "
            "Matters list: sortable columns. Add New Matter button. "
            "Clients list: Add New Client button (name, email, status, matters). "
            "Calendar: Add event button, date/time/label/event_type fields. "
            "Settings: Change user name option. "
            "Sort by column: click column header or sort button."
        ),
        "AutoMail": (
            "SITE: Webmail client. NAV: Inbox, Drafts, Sent, Spam, Templates folder tabs. "
            "Email list: shows from_email, subject, date, is_starred, is_important flags. "
            "Actions per email: Star (⭐), Archive, Mark as Spam, Delete (🗑), Forward, Reply. "
            "Select email: checkbox. Select all: top checkbox. Clear selection: deselect all. "
            "Important: click flag/important icon. "
            "Templates tab: list of templates with template_name, subject, to fields. "
            "Template actions: Select (use it), Send, Save as Draft. "
            "Pagination: Next/Previous page arrows at bottom of email list."
        ),
        "AutoDelivery": (
            "SITE: Food delivery app. NAV: Restaurants list, Cart, Orders. "
            "Restaurant cards: name, cuisine, rating, description. Click to view restaurant detail. "
            "Restaurant detail: menu items with size, price, quantity selector. Add to cart. "
            "Cart page: shows items with preferences (dietary), size, quantity, price, restaurant name. "
            "Cart: Dropoff preference selector (Hand it to me / Leave at door / Text when arriving). "
            "Delivery priority: Normal/Priority/Scheduled option. "
            "Checkout: proceed to checkout button. "
            "Pagination: next/previous page for restaurant list. "
            "View all restaurants: click All Restaurants or similar nav link."
        ),
        "AutoLodge": (
            "SITE: Hotel/lodging booking (Airbnb-style). Shows listing cards. "
            "Listing card/detail: title, host_name, location, price/night, rating, reviews count, amenities list, guests. "
            "Guest selector: +/- buttons or dropdown to set number of guests. "
            "Actions: Reserve/Book Now → payment form, Add to Wishlist, Submit Review. "
            "Payment form fields: card_number, expiration (MM/YY), cvv, zipcode, country. "
            "Payment methods: Credit card, PayPal, Bank transfer. "
            "Search: search bar for hotel name/location. "
            "Filters: rating, region/country, price range. "
            "Review form: rating stars + text area."
        ),
        "AutoConnect": (
            "SITE: Professional network (LinkedIn-style). NAV: Feed, Jobs, People, Company Pages. "
            "Feed: posts with text, author, Like/Comment buttons. Comment: text field + submit. "
            "Jobs section: job listings with title, company, location, Apply button. "
            "My Applications: list with status (Pending/Accepted/Rejected), Withdraw/Cancel button. "
            "Company Pages: Follow/Unfollow button on each page. "
            "People/Users: search bar for users. "
            "Profile: Edit profile (bio, skills, photo). "
            "Back to Jobs: breadcrumb or Back button from job detail."
        ),
        "AutoHire": (
            "SITE: Freelancer hiring platform. NAV: Jobs, Hires, Experts/Browse. "
            "Expert/Consultant cards: name, role, country, rating, price. "
            "Expert actions: Hire Now button, Hire Later button, View Profile. "
            "Hire Later page: list of saved experts with Remove button. "
            "Job Posting: Post a Job / + button → form with title, description, rate_from, rate_to, project size. "
            "Job posting form: title field, description, rate range, project size (Small/Medium/Large). "
            "Close job posting window: X/Cancel button on job posting modal. "
            "Search skills: search bar for skills. "
            "NAV: Jobs link, Hires link in navbar. "
            "Hiring Team: section showing team members."
        ),
        "AutoCalendar": (
            "SITE: Calendar app (Google Calendar-style). "
            "View buttons: Day, 5-day (work week), Week, Month — click to switch view. "
            "Left sidebar: list of calendars with + button to add new calendar. "
            "Add New Calendar modal: name + description fields. "
            "Events: click on time slot or + button to add event. "
            "Event form: title, date, time, visibility (Public/Private/Default), reminders (minutes), "
            "meeting_link, attendees (email), all_day toggle, recurrence, calendar, description, busy. "
            "Event actions: Edit, Delete. "
            "Attendees: add attendee email field in event edit form."
        ),
        "AutoList": (
            "SITE: Task management (Trello/Monday-style). "
            "Tasks list: each task has name, description, date, priority (1=High/2=Medium/3=Low), status. "
            "Add Task button: + or 'Add Task' button to create new task. "
            "Task actions: Edit (pencil icon) → modal, Delete (trash icon) → confirm. "
            "Edit task modal: name, description, date, priority fields. "
            "Team tab/section: list of team members with name, role. "
            "Add member: search by name and add. "
            "Assign role: dropdown next to member name."
        ),
        "AutoRide": (
            "SITE: Ride-sharing app (Uber-style). "
            "Main page: Location (pickup) input + Destination input fields. "
            "Available rides: list with ride_name, price, estimated time, scheduled. "
            "Reserve button on each ride card. "
            "Date picker: select trip date (calendar widget). "
            "Time picker: select trip time. "
            "Search location: type in the search/destination input box. "
            "Reservation history: list of upcoming/past rides with Cancel button. "
            "Cancel: click Cancel on a specific reservation. "
            "Next pickup: shows scheduled pickup details."
        ),
        "AutoMedic": (
            "SITE: Medical/healthcare platform. NAV: Doctors, Appointments, Medical Records/Analysis. "
            "Doctor cards: doctor_name, speciality, rating, consultation_fee, language. "
            "Doctor card actions: View Profile, Book Appointment, Contact Doctor. "
            "Doctor profile detail: full info, education section. "
            "Appointment form: doctor_name, speciality, date, time fields. "
            "Quick appointment form: speciality, patient_name, patient_email. "
            "Medical Records/Analysis: searchable list with record_title, doctor_name, record_type, date. "
            "Search: filter fields for doctor_name, speciality, record_title, etc. "
            "Contact doctor form: opens when Contact button clicked on doctor card."
        ),
    }
    return ctx.get(website, "")


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

    # Website detection
    website_name = _detect_website(url)
    website_ctx = _website_context(website_name)

    # Parse ALL constraints from the task
    task_constraints = _parse_task_constraints(task)
    constraints_block = _format_constraints_block(task_constraints)

    # Extract credentials from both task text and relevant_data
    creds_from_task = _extract_credentials_from_task(task)
    creds_from_data: Dict[str, str] = {}
    if relevant_data and isinstance(relevant_data, dict):
        for k, v in relevant_data.items():
            if isinstance(v, str) and v:
                creds_from_data[str(k)] = str(v)
    # Also add all "equals" constraints as directly usable field values
    for c in task_constraints:
        if c["op"] == "equals" and isinstance(c["value"], str):
            field = c["field"]
            if field not in creds_from_task and field not in creds_from_data:
                creds_from_task[field] = c["value"]
    all_creds = {**creds_from_task, **creds_from_data}

    system_msg = (
        "You are a web automation agent. Return JSON only (no markdown). "
        "Keys: action, candidate_id, text, url, evaluation_previous_goal, memory, next_goal.\n"
        "action: click|type|select|navigate|scroll_down|scroll_up|done. "
        "click/type/select: candidate_id=integer from BROWSER_STATE. "
        "navigate: url=full URL (keep ?seed=X param). "
        "done: only when task is fully completed.\n"
        "RULES: Copy values EXACTLY from TASK_CREDENTIALS/TASK_CONSTRAINTS (include trailing spaces). "
        "equals→type exact value. not_equals→use any OTHER value. contains→find item with that substring. "
        "not_contains/not_in→find item WITHOUT that value. greater/less→numeric comparison.\n"
        "CREDENTIALS: username/email may have trailing spaces - type them exactly as shown in quotes. "
        "job_title_contains→type any title CONTAINING that substring.\n"
        "MULTI-STEP: complete login first, then the secondary action. Track progress in memory.\n"
        "TOOLS: Return {\"tool\":\"<name>\",\"args\":{...}} to inspect page. Max 1 tool per step. "
        "Tools: list_cards({max_cards?,max_text?}); search_text({query}); list_links({}); extract_forms({})."
    )

    history_lines: List[str] = []
    for h in (history or [])[-2:]:
        step = h.get("step", "?")
        action = h.get("action", "")
        cid = h.get("candidate_id")
        text = str(h.get("text", ""))[:60]
        ok = h.get("exec_ok", True)
        err = h.get("error")
        suffix = "OK" if ok else f"FAIL:{str(err)[:40]}"
        history_lines.append(f"{step}.{action} cid={cid} t={text} [{suffix}]")

    hint = _history_hint(history)

    structured = _structured_hints(task, candidates)

    # Only include cards on early steps to save tokens
    cards_preview = ""
    if int(step_index) <= 2:
        try:
            cards_obj = _tool_list_cards(candidates=candidates, max_cards=6, max_text=120, max_actions_per_card=1)
            if isinstance(cards_obj, dict) and cards_obj.get("ok") and cards_obj.get("cards"):
                cards_preview = json.dumps(cards_obj.get("cards"), ensure_ascii=True)
                if len(cards_preview) > 600:
                    cards_preview = cards_preview[:597] + "..."
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
        creds_block = "TASK_CREDENTIALS (use EXACTLY as-is, no modifications - including spaces):\n"
        for k, v in all_creds.items():
            # Show the value in quotes so trailing/leading spaces are visible
            creds_block += f"  {k}: '{v}'\n"

    # Cap large sections to control per-step token costs
    page_summary_capped = page_summary[:400] if page_summary else ""
    # DOM digest only on step 0 (first look)
    dom_digest_capped = dom_digest[:200] if dom_digest and int(step_index) == 0 else ""
    structured_str = json.dumps(structured, ensure_ascii=True)
    if len(structured_str) > 500:
        structured_str = structured_str[:497] + "..."
    website_ctx_short = (website_ctx[:150] + "...") if len(website_ctx) > 150 else website_ctx
    playbook_capped = (playbook[:350] + "...") if len(playbook) > 350 else playbook

    user_msg = (
        f"TASK: {task}\n"
        f"TYPE:{task_type} SITE:{website_name} STEP:{int(step_index)} URL:{url}\n\n"
        + (f"SITE_HINTS: {website_ctx_short}\n\n" if website_ctx_short else "")
        + (creds_block + "\n" if creds_block else "")
        + (constraints_block + "\n\n" if constraints_block else "")
        + f"{playbook_capped}\n\n"
        + f"PAGE:\n{page_summary_capped}\n\n"
        + (f"DOM:\n{dom_digest_capped}\n\n" if dom_digest_capped else "")
        + (f"CARDS:\n{cards_preview}\n\n" if cards_preview else "")
        + f"STATE:\n{structured_str}\n\n"
        + (f"HISTORY:\n{chr(10).join(history_lines)}\n\n" if history_lines else "")
        + (f"HINT: {extra_hint}\n\n" if extra_hint else "")
        + (f"MEMORY:\n{agent_mem}\n" if agent_mem else "")
        + (f"DELTA: {str(state_delta)[:200]}\n\n" if state_delta else "")
        + "BROWSER_STATE:\n" + browser_state + "\n\n"
        + "ONE JSON action only."
    )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "250"))

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

    # Hard step cap: force done after 7 steps to avoid over-cost
    max_steps_hard = int(os.getenv("AGENT_MAX_STEPS", "7"))
    if step_index >= max_steps_hard:
        return _resp([{"type": "DoneAction", "success": True}], {"decision": "forced_done_step_cap", "step_index": step_index})

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
