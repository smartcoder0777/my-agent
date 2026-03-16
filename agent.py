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

    for field_pat, key in patterns:
        # Check it's a proper equals (not "not equals")
        m = re.search(
            r"(?<!\w)" + field_pat + r"\s+equals?\s+['\"]?([^\s,'\"\n\]]+)['\"]?",
            t, re.IGNORECASE
        )
        if m:
            # Verify not preceded by "not"
            prefix = t[max(0, m.start()-5):m.start()].lower()
            if "not" not in prefix:
                val = m.group(1).strip().rstrip(".,;:")
                if val and key not in creds:
                    creds[key] = val

    # Special: "writing a title of job for 'X'"
    m = re.search(r"writing\s+a\s+title\s+of\s+job\s+for\s+['\"]([^'\"]+)['\"]", t, re.IGNORECASE)
    if m:
        creds["job_title"] = m.group(1).strip()

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
    if re.search(r"select\s+(a\s+)?time\s+for\s+(my\s+|your\s+)?booking", t, re.IGNORECASE):
        return "SELECT_TIME"
    if re.search(r"next\s+pickup", t, re.IGNORECASE):
        return "NEXT_PICKUP"

    # ---- AutoMail (8005) ----
    if re.search(r"mark\s+as\s+spam", t, re.IGNORECASE):
        return "EMAIL_MARK_SPAM"
    if re.search(r"(mark|move)\s+.*(spam|junk)", t, re.IGNORECASE):
        return "EMAIL_MARK_SPAM"
    if re.search(r"star\s+the\s+email", t, re.IGNORECASE):
        return "STAR_AN_EMAIL"
    if re.search(r"archive\s+the\s+email", t, re.IGNORECASE):
        return "ARCHIVE_EMAIL"
    if re.search(r"delete\s+the\s+email", t, re.IGNORECASE):
        return "DELETE_EMAIL"
    if re.search(r"forward\s+the\s+email", t, re.IGNORECASE):
        return "FORWARD_EMAIL"
    if re.search(r"mark.*email.*important", t, re.IGNORECASE):
        return "MARK_EMAIL_AS_IMPORTANT"
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
    if re.search(r"switch\s+to\s+5.?day\s+view", t, re.IGNORECASE):
        return "SELECT_FIVE_DAYS"
    if re.search(r"(add\s+|click.*)\s*add\s+calendar\s+button", t, re.IGNORECASE):
        return "ADD_NEW_CALENDAR"
    if re.search(r"create\s+a\s+new\s+calendar", t, re.IGNORECASE):
        return "CREATE_CALENDAR"
    if re.search(r"add\s+an?\s+attendee\s+to\s+the\s+event", t, re.IGNORECASE):
        return "EVENT_ADD_ATTENDEE"
    if re.search(r"delete\s+an?\s+added\s+event", t, re.IGNORECASE):
        return "DELETE_ADDED_EVENT"
    if re.search(r"cancel\s+an?\s+event", t, re.IGNORECASE):
        return "CANCEL_ADD_EVENT"
    if re.search(r"add\s+a\s+new\s+calendar\s+event", t, re.IGNORECASE):
        return "NEW_CALENDAR_EVENT_ADDED"
    if re.search(r"add\s+an?\s+event\b", t, re.IGNORECASE):
        return "ADD_EVENT"
    if re.search(r"(show|view)\s+.*pending\s+events", t, re.IGNORECASE):
        return "VIEW_PENDING_EVENTS"

    # ---- AutoList (8011) ----
    if re.search(r"add\s+members?\s+to\s+the\s+team", t, re.IGNORECASE):
        return "AUTOLIST_TEAM_MEMBERS_ADDED"
    if re.search(r"assign\s+a\s+role\s+.*team\s+member", t, re.IGNORECASE):
        return "AUTOLIST_TEAM_ROLE_ASSIGNED"
    if re.search(r"edit\s+task\s+modal\s+open", t, re.IGNORECASE):
        return "AUTOLIST_EDIT_TASK_MODAL_OPENED"
    if re.search(r"button\s+to\s+add\s+a\s+task\s+is\s+clicked", t, re.IGNORECASE):
        return "AUTOLIST_ADD_TASK_CLICKED"
    if re.search(r"add\s+a\s+task\s+where", t, re.IGNORECASE):
        return "AUTOLIST_TASK_ADDED"

    # ---- AutoMedic (8013) ----
    if re.search(r"show\s+details\s+for\s+a\s+doctor", t, re.IGNORECASE):
        return "VIEW_DOCTOR_PROFILE"
    if re.search(r"show\s+me\s+information\s+about\s+doctors", t, re.IGNORECASE):
        return "SEARCH_DOCTORS"
    if re.search(r"(search|retrieve)\s+(medical|details of medical)", t, re.IGNORECASE):
        return "SEARCH_MEDICAL_ANALYSIS"
    if re.search(r"view\s+medical\s+analysis", t, re.IGNORECASE):
        return "VIEW_MEDICAL_ANALYSIS"
    if re.search(r"open\s+appointment\s+form", t, re.IGNORECASE):
        return "OPEN_APPOINTMENT_FORM"
    if re.search(r"open\s+contact\s+doctor\s+form", t, re.IGNORECASE):
        return "OPEN_CONTACT_DOCTOR_FORM"
    if re.search(r"contact\s+(a\s+)?doctor", t, re.IGNORECASE):
        return "CONTACT_DOCTOR"
    if re.search(r"retrieve\s+details\s+of\s+appointments", t, re.IGNORECASE):
        return "SEARCH_APPOINTMENT"
    if re.search(r"request\s+a\s+quick\s+appointment", t, re.IGNORECASE):
        return "REQUEST_QUICK_APPOINTMENT"
    if re.search(r"doctor.*education", t, re.IGNORECASE):
        return "VIEW_DOCTOR_EDUCATION"

    # ---- AutoConnect (8008) ----
    if re.search(r"comment\s+on\s+the\s+post", t, re.IGNORECASE):
        return "COMMENT_ON_POST"
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
    if re.search(r"(job\s+posting|writing\s+a\s+title\s+of\s+job)", t, re.IGNORECASE):
        return "JOB_POSTING"
    if re.search(r"edit\s+profile\s+(location|email)", t, re.IGNORECASE):
        if "location" in t:
            return "EDIT_PROFILE_LOCATION"
        return "EDIT_PROFILE_EMAIL"

    # ---- AutoLodge (8007) ----
    if re.search(r"confirm\s+the\s+booking", t, re.IGNORECASE):
        return "BOOKING_CONFIRM"
    if re.search(r"set\s+the\s+number\s+of\s+guests", t, re.IGNORECASE):
        return "EDIT_NUMBER_OF_GUESTS"
    if re.search(r"(open\s+)?guest\s+selector\s+dropdown", t, re.IGNORECASE):
        return "PEOPLE_DROPDOWN_OPENED"
    if re.search(r"select\s+(a\s+)?payment\s+method", t, re.IGNORECASE):
        return "PAYMENT_METHOD_SELECTED"
    if re.search(r"(reserve|book)\s+the\s+hotel", t, re.IGNORECASE):
        return "RESERVE_HOTEL"
    if re.search(r"search\s+for\s+hotels?", t, re.IGNORECASE):
        return "SEARCH_HOTEL"
    if re.search(r"submit\s+a\s+review", t, re.IGNORECASE):
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
    if re.search(r"search\s+for\s+restaurants?\s+(where|that)", t, re.IGNORECASE):
        return "SEARCH_DELIVERY_RESTAURANT"
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
    if re.search(r"sort\s+matters?\s+so\s+that", t, re.IGNORECASE):
        return "SORT_MATTER_BY_CREATED_AT"
    if re.search(r"change\s+(user\s+)?name\s+to", t, re.IGNORECASE):
        return "CHANGE_USER_NAME"
    if re.search(r"show.*pending\s+events\s+on\s+the\s+calendar", t, re.IGNORECASE):
        return "VIEW_PENDING_EVENTS"

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
        "PLAYBOOK: 1) Find the doctor matching constraints. 2) View their profile. "
        "3) Scroll to/click the Education section on their profile."
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
        "- Copy ALL values EXACTLY as provided in TASK_CREDENTIALS or TASK_CONSTRAINTS 'equals' fields\n"
        "- Do NOT correct typos, do NOT remove numbers, do NOT truncate strings\n"
        "- Example: if task says 'Sofia 4', type 'Sofia 4' not 'Sofia'\n"
        "\n"
        "CONSTRAINT HANDLING:\n"
        "- TASK_CONSTRAINTS lists ALL field constraints for this task\n"
        "- 'equals' → type/select EXACTLY that value\n"
        "- 'not_equals' → choose ANY valid value DIFFERENT from the listed one\n"
        "  (e.g. card_number not '5500000000000004' → use '4111111111111111')\n"
        "  (e.g. expiration not '06/26' → use '12/27')\n"
        "  (e.g. destination not 'Business Tower...' → type any other real address)\n"
        "- 'contains' → the item's field value must include that substring (use to identify the item)\n"
        "- 'not_contains' → the item's field value must NOT include that substring\n"
        "- 'greater_than'/'less_than' → numeric or date comparison (use to identify/filter items)\n"
        "- 'not_in' → the item's field must NOT be any of the listed values\n"
        "\n"
        "FINDING THE RIGHT ITEM:\n"
        "- When a task has multiple constraints, use list_cards or search_text to browse items\n"
        "- Check ALL constraints against each candidate item before selecting it\n"
        "- For email tasks: match subject AND sender constraints simultaneously\n"
        "- For task management: match name, description, date, AND priority constraints\n"
        "- For booking: match rating, price, location, host, reviews, amenities, title constraints\n"
        "\n"
        "CREDENTIAL HANDLING:\n"
        "- TASK_CREDENTIALS contains extracted field values (credentials + form fields)\n"
        "- Map 'username'/'signup_username' → username input\n"
        "- Map 'email'/'signup_email' → email input\n"
        "- Map 'password'/'signup_password' → password input\n"
        "- Map 'cvv' → CVV/security code input\n"
        "- Map 'zipcode' → ZIP/postal code input\n"
        "- Map 'country' → country selector/input\n"
        "- Map 'job_title' → job title input\n"
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
        f"WEBSITE: {website_name}\n"
        f"STEP: {int(step_index)}\n"
        f"URL: {url}\n\n"
        + (f"WEBSITE_CONTEXT:\n{website_ctx}\n\n" if website_ctx else "")
        + (creds_block + "\n" if creds_block else "")
        + (constraints_block + "\n\n" if constraints_block else "")
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
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "600"))

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
