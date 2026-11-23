"""Turn Alpaca's markdown doc pages into search chunks.

Guide pages are normal markdown, so they get split by heading. API reference
pages are mostly one big OpenAPI JSON block. Searching raw JSON works badly,
so each endpoint gets rewritten as plain text first ("POST /v2/orders takes
symbol, qty, ..."). The same parsed endpoints also feed the call checker.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import CHUNK_CHARS


@dataclass
class Page:
    url: str
    slug: str
    title: str
    kind: str  # "guide" or "reference"
    body: str  # markdown without the front matter


@dataclass
class Chunk:
    id: str
    text: str
    slug: str
    url: str
    title: str
    section: str


# ---------------------------------------------------------------- pages


def page_slug(url: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return slug[:-3] if slug.endswith(".md") else slug


def strip_front_matter(markdown: str) -> str:
    if markdown.startswith("---"):
        end = markdown.find("\n---", 3)
        if end != -1:
            return markdown[end + 4 :].lstrip("\n")
    return markdown


def parse_page(url: str, markdown: str) -> Page:
    body = strip_front_matter(markdown)
    match = re.search(r"^# (.+)$", body, re.M)
    title = match.group(1).strip() if match else page_slug(url)
    kind = "reference" if "/reference/" in url else "guide"
    return Page(url=url, slug=page_slug(url), title=title, kind=kind, body=body)


# ---------------------------------------------------------- guide pages

_KNOWN_TAGS = r"(?:table|thead|tbody|tr|th|td|b|i|em|strong|br|p|div|span|img|Callout|HTMLBlock|Image)"


def _strip_tags(text: str) -> str:
    return re.sub(rf"</?{_KNOWN_TAGS}\b[^>]*>", "", text)


def _html_table_to_rows(match: re.Match) -> str:
    rows = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", match.group(0), re.S):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)
        cells = [" ".join(_strip_tags(c).split()) for c in cells]
        if cells:
            rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def clean_guide_markdown(body: str) -> str:
    """Remove the CSS and HTML noise that ReadMe pages carry."""
    text = re.sub(r"<style>.*?</style>", "", body, flags=re.S)
    text = re.sub(r"<table>.*?</table>", _html_table_to_rows, text, flags=re.S)
    text = text.replace("<HTMLBlock>{`", "").replace("`}</HTMLBlock>", "")
    text = _strip_tags(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long(text: str, limit: int) -> list[str]:
    """Split text into pieces no longer than limit, preferring line breaks."""
    pieces, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            pieces.append(current)
            current = line
        else:
            current = candidate
    if current.strip():
        pieces.append(current)
    return [p for p in pieces if p.strip()]


def _paragraphs(text: str) -> list[tuple[str, bool]]:
    """Split markdown into (paragraph, starts_with_heading) pairs.

    Blank lines and "# comment" lines inside ``` code fences are part of the
    code, not paragraph breaks or headings.
    """
    paras: list[tuple[str, bool]] = []
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        if current and "\n".join(current).strip():
            first = current[0]
            is_heading = bool(re.match(r"^#{1,6}\s+\S", first)) and not first.startswith("```")
            paras.append(("\n".join(current).strip(), is_heading))
        current.clear()

    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            current.append(line)
            continue
        if not in_fence and not line.strip():
            flush()
            continue
        if not in_fence and re.match(r"^#{1,6}\s+\S", line) and current:
            flush()  # a heading always starts a new paragraph
        current.append(line)
    flush()
    return paras


def chunk_guide(page: Page, limit: int = CHUNK_CHARS) -> list[Chunk]:
    text = clean_guide_markdown(page.body)
    # Walk the page paragraph by paragraph, remembering the latest headings.
    headings: dict[int, str] = {}
    blocks: list[tuple[str, str]] = []  # (section label, paragraph)
    for para, is_heading in _paragraphs(text):
        heading = re.match(r"^(#{1,6})\s+(.+)$", para.split("\n", 1)[0]) if is_heading else None
        if heading:
            level = len(heading.group(1))
            headings = {lvl: h for lvl, h in headings.items() if lvl < level}
            headings[level] = heading.group(2).strip()
        section = " > ".join(headings[lvl] for lvl in sorted(headings) if headings[lvl] != page.title)
        blocks.append((section or page.title, para))

    chunks: list[Chunk] = []
    current, current_section = "", ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            header = f"{page.title} > {current_section}" if current_section != page.title else page.title
            chunks.append(
                Chunk(
                    id=f"{page.slug}-{len(chunks):03d}",
                    text=f"{header}\n\n{current.strip()}",
                    slug=page.slug,
                    url=page.url,
                    title=page.title,
                    section=current_section,
                )
            )
        current = ""

    for section, para in blocks:
        # Start a new chunk when the top-level section changes, so a chunk
        # doesn't mix two unrelated topics.
        top_changed = current and section.split(" > ")[0] != current_section.split(" > ")[0]
        if top_changed or len(current) + len(para) + 2 > limit:
            flush()
        if not current:
            current_section = section
        if len(para) > limit:
            for piece in _split_long(para, limit):
                current = piece
                flush()
                current_section = section
        else:
            current = f"{current}\n\n{para}" if current else para
    flush()
    return chunks


# ------------------------------------------------------ reference pages


def extract_spec(body: str) -> dict | None:
    match = re.search(r"```json\n(.*?)\n```", body, re.S)
    return json.loads(match.group(1)) if match else None


class _Resolver:
    """Follows "$ref" pointers and merges "allOf" in an OpenAPI spec."""

    def __init__(self, spec: dict):
        self.spec = spec

    def ref(self, node, depth: int = 0):
        while isinstance(node, dict) and "$ref" in node and depth < 20:
            target = self.spec
            for part in node["$ref"].lstrip("#/").split("/"):
                target = target[part]
            node = target
            depth += 1
        return node

    def flat(self, schema, depth: int = 0) -> dict:
        schema = self.ref(schema) or {}
        if depth > 6 or not isinstance(schema, dict):
            return {}
        if "allOf" in schema:
            merged = {k: v for k, v in schema.items() if k != "allOf"}
            for part in schema["allOf"]:
                part = self.flat(part, depth + 1)
                for key, value in part.items():
                    if key == "properties":
                        merged.setdefault("properties", {}).update(value)
                    elif key == "required":
                        merged["required"] = sorted(set(merged.get("required", [])) | set(value))
                    else:
                        merged.setdefault(key, value)
            schema = merged
        return schema


def _clean(text: str | None) -> str:
    text = (text or "").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def _field(resolver: _Resolver, schema, depth: int = 0) -> dict:
    schema = resolver.flat(schema)
    types = schema.get("type")
    if not types:
        options = schema.get("oneOf") or schema.get("anyOf") or []
        types = sorted({resolver.flat(o).get("type") for o in options if resolver.flat(o).get("type")}) or None
    field = {"type": types, "enum": schema.get("enum"), "description": _clean(schema.get("description"))}
    if depth < 2 and schema.get("properties"):
        field["required"] = schema.get("required", [])
        field["properties"] = {
            name: _field(resolver, sub, depth + 1) for name, sub in schema["properties"].items()
        }
    return field


def spec_to_endpoints(spec: dict, doc_url: str) -> list[dict]:
    resolver = _Resolver(spec)
    endpoints = []
    for path, item in spec.get("paths", {}).items():
        common = item.get("parameters", [])
        for method in ("get", "post", "put", "patch", "delete"):
            op = item.get(method)
            if not op:
                continue
            params = []
            for param in common + op.get("parameters", []):
                param = resolver.ref(param)
                schema = resolver.flat(param.get("schema", {}))
                params.append(
                    {
                        "name": param["name"],
                        "in": param["in"],
                        "required": bool(param.get("required")),
                        "type": schema.get("type"),
                        "enum": schema.get("enum"),
                        "description": _clean(param.get("description") or schema.get("description")),
                    }
                )
            body = None
            if op.get("requestBody"):
                content = resolver.ref(op["requestBody"]).get("content", {})
                schema = content.get("application/json", {}).get("schema")
                if schema:
                    body = _field(resolver, schema)
            response_fields = []
            ok = resolver.ref(op.get("responses", {}).get("200", {}))
            if ok:
                schema = resolver.flat(ok.get("content", {}).get("application/json", {}).get("schema", {}))
                if schema.get("type") == "array":
                    schema = resolver.flat(schema.get("items", {}))
                for name, sub in schema.get("properties", {}).items():
                    sub = resolver.flat(sub)
                    response_fields.append(
                        {"name": name, "type": sub.get("type"), "description": _clean(sub.get("description"))[:200]}
                    )
            endpoints.append(
                {
                    "method": method.upper(),
                    "path": path,
                    "summary": op.get("summary", ""),
                    "description": _clean(op.get("description")),
                    "params": params,
                    "body": body,
                    "response_fields": response_fields,
                    "doc_url": doc_url,
                }
            )
    return endpoints


def _type_label(field: dict) -> str:
    kind = field.get("type")
    if isinstance(kind, list):
        kind = " or ".join(str(k) for k in kind)
    return kind or "any"


def _indent(text: str, prefix: str = "  ") -> str:
    return text.replace("\n", "\n" + prefix)


def endpoint_to_text(ep: dict) -> str:
    lines = [f"{ep['method']} {ep['path']} ({ep['summary']})"]
    if ep["description"]:
        lines.append(ep["description"])
    for where, label in (("path", "Path parameters"), ("query", "Query parameters")):
        params = [p for p in ep["params"] if p["in"] == where]
        if params:
            lines.append(f"{label}:")
            for p in params:
                extra = ", required" if p["required"] else ", optional"
                if p["enum"]:
                    extra += ", one of: " + ", ".join(map(str, p["enum"]))
                lines.append(f"- {p['name']} ({_type_label(p)}{extra}): {_indent(p['description'])}")
    body = ep.get("body")
    if body and body.get("properties"):
        required = set(body.get("required", []))
        lines.append("Request body fields:")
        for name, field in body["properties"].items():
            extra = ", required" if name in required else ""
            if field.get("enum"):
                extra += ", one of: " + ", ".join(str(e) for e in field["enum"] if e != "")
            lines.append(f"- {name} ({_type_label(field)}{extra}): {_indent(field['description'])}")
            for sub_name, sub in (field.get("properties") or {}).items():
                lines.append(f"  - {name}.{sub_name} ({_type_label(sub)}): {_indent(sub['description'], '    ')}")
    if ep["response_fields"]:
        lines.append("Response fields:")
        for f in ep["response_fields"]:
            desc = f": {f['description']}" if f["description"] else ""
            lines.append(f"- {f['name']} ({_type_label(f)}){_indent(desc)}")
    return "\n".join(lines)


def chunk_reference(page: Page, endpoints: list[dict], limit: int = CHUNK_CHARS) -> list[Chunk]:
    chunks = []
    for ep in endpoints:
        header = f"{ep['method']} {ep['path']} ({ep['summary']})"
        body = endpoint_to_text(ep).split("\n", 1)[1] if "\n" in endpoint_to_text(ep) else ""
        pieces = _split_long(body, limit - len(header) - 20) or [""]
        for n, piece in enumerate(pieces, start=1):
            label = header if len(pieces) == 1 else f"{header}, part {n} of {len(pieces)}"
            chunks.append(
                Chunk(
                    id=f"{page.slug}-{len(chunks):03d}",
                    text=f"{label}\n\n{piece.strip()}".strip(),
                    slug=page.slug,
                    url=page.url,
                    title=page.title,
                    section=label,
                )
            )
    return chunks


def chunk_page(page: Page) -> tuple[list[Chunk], list[dict]]:
    """Return (chunks, endpoints) for one page. Guides have no endpoints."""
    if page.kind == "reference":
        spec = extract_spec(page.body)
        if spec:
            endpoints = spec_to_endpoints(spec, page.url)
            return chunk_reference(page, endpoints), endpoints
    return chunk_guide(page), []
