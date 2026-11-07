"""
Context Builder and Entity Resolution Engine for EOS LLM.

Inspired by Open-Jev tabular data handling and Laya cross-platform entity resolution:
- Normalizes heterogeneous structured inputs: JSON, YAML, CSV tabular records, entities.
- Supports multimodal feature tags: <|audio|> and <|vision|> descriptors.
- Formats structured payloads into deterministic <|state|> representations ready for
  the Rhen Delimiter Packer and Three-Layer Pipeline.
"""

import csv
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

#: Longest section name rendered in a section header line.
MAX_SECTION_NAME_CHARS: int = 80

#: Line breaks that would start a new physical line inside a rendered row or header.
_LINE_BREAK_RE = re.compile(r"\r\n|[\r\n\v\f\x1c-\x1e\x85\N{LINE SEPARATOR}\N{PARAGRAPH SEPARATOR}]")

#: A run of backslashes directly before a pipe; the pipe escape must not be swallowed by them.
_PIPE_RE = re.compile(r"(\\*)\|")

#: A line that starts with dashes would read as a builder section header such as "--- ENTITIES ---".
_HEADER_LIKE_RE = re.compile(r"(?m)^([ \t]*)---")


def _neutralize(value: Any) -> str:
    """Neutralize control-token syntax and header-like lines in user-provided text.

    Local copy of the rule the Rhen packer applies (``<|`` becomes ``<¦`` and ``|>``
    becomes ``¦>``), so the builder does not import the packer. Lines that start with
    ``---`` are rewritten as ``- - -`` so user text cannot forge a section header. The
    builder's own ``<|audio|>`` and ``<|vision|>`` tags and section headers are added
    after this step and stay intact.
    """
    text = str(value).replace("<|", "<¦").replace("|>", "¦>")
    return _HEADER_LIKE_RE.sub(lambda m: m.group(1) + "- - -", text)


def _table_cell(value: Any) -> str:
    """Render one Markdown table cell on a single line, with every pipe escaped."""
    text = _LINE_BREAK_RE.sub(" ", _neutralize(value))
    return _PIPE_RE.sub(lambda m: m.group(1) * 2 + "\\|", text)


def _section_title(name: Optional[str], default: str) -> str:
    """Section name as rendered in a header: neutralized, one line, at most 80 characters."""
    title = _LINE_BREAK_RE.sub(" ", _neutralize(name if name else default)).strip()
    return (title or default)[:MAX_SECTION_NAME_CHARS]


class ContextBuilder:
    """Builder for constructing rich, structured state context."""

    def __init__(self):
        self.sections: List[Dict[str, str]] = []
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.multimodal_items: List[Dict[str, Any]] = []

    def add_text(self, text: str, section_name: Optional[str] = None) -> "ContextBuilder":
        """Add a raw text section. Content is neutralized; the section name is one line, capped at 80 characters."""
        self.sections.append({
            "type": "text",
            "name": _section_title(section_name, "Context"),
            "content": _neutralize(text).strip(),
        })
        return self

    def add_json(self, data: Union[Dict, List], section_name: Optional[str] = None) -> "ContextBuilder":
        """Add formatted JSON data. The rendered text is neutralized, so string values and keys are covered."""
        # Control-token syntax can only occur inside JSON strings (no JSON structure character
        # contains '<' or '|'), so neutralizing the rendered text touches only user strings.
        json_str = _neutralize(json.dumps(data, indent=2, sort_keys=True))
        self.sections.append({
            "type": "json",
            "name": _section_title(section_name, "JSON Data"),
            "content": json_str,
        })
        return self

    def add_yaml(self, yaml_content: str, section_name: Optional[str] = None) -> "ContextBuilder":
        """Add YAML configuration or specification text."""
        self.sections.append({
            "type": "yaml",
            "name": _section_title(section_name, "YAML Specification"),
            "content": _neutralize(yaml_content).strip(),
        })
        return self

    def add_tabular(
        self,
        rows: List[Dict[str, Any]],
        headers: Optional[List[str]] = None,
        section_name: Optional[str] = None,
    ) -> "ContextBuilder":
        """Add tabular records formatted as Markdown / Open-Jev tabular representation."""
        if not rows:
            return self

        if not headers:
            headers = list(rows[0].keys())

        # Construct markdown table; every cell stays on one line with its pipes escaped.
        header_line = "| " + " | ".join(_table_cell(h) for h in headers) + " |"
        sep_line = "| " + " | ".join("---" for _ in headers) + " |"
        row_lines = []
        for r in rows:
            row_lines.append("| " + " | ".join(_table_cell(r.get(h, "")) for h in headers) + " |")

        table_str = "\n".join([header_line, sep_line] + row_lines)
        self.sections.append({
            "type": "tabular",
            "name": _section_title(section_name, "Tabular Records"),
            "content": table_str,
        })
        return self

    def add_csv(self, csv_str: str, section_name: Optional[str] = None) -> "ContextBuilder":
        """Parse raw CSV text and store as tabular records."""
        f = io.StringIO(csv_str.strip())
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames or []
        return self.add_tabular(rows, headers=list(headers), section_name=section_name or "CSV Data")

    def add_entity(
        self,
        entity_type: str,
        entity_id: str,
        attributes: Dict[str, Any],
    ) -> "ContextBuilder":
        """Register an entity with resolved attributes (Laya entity pattern)."""
        key = f"{entity_type}:{entity_id}"
        self.entities[key] = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "attributes": attributes,
        }
        return self

    def add_audio_descriptor(
        self,
        summary: str,
        duration_sec: Optional[float] = None,
        sample_rate: Optional[int] = None,
        detected_events: Optional[List[str]] = None,
    ) -> "ContextBuilder":
        """Add an acoustic / audio modality descriptor."""
        descriptor = {
            "type": "audio",
            "summary": summary,
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
            "detected_events": detected_events or [],
        }
        self.multimodal_items.append(descriptor)
        return self

    def add_vision_descriptor(
        self,
        summary: str,
        resolution: Optional[Tuple[int, int]] = None,
        objects_detected: Optional[List[str]] = None,
    ) -> "ContextBuilder":
        """Add a visual scene / vision modality descriptor."""
        descriptor = {
            "type": "vision",
            "summary": summary,
            "resolution": resolution,
            "objects_detected": objects_detected or [],
        }
        self.multimodal_items.append(descriptor)
        return self

    def build_state_string(self) -> str:
        """Serialize complete context into a single normalized state string for Rhen."""
        parts = []

        # 1. Registered Entities (if any). Entity fields are user text: neutralized at render time.
        if self.entities:
            parts.append("--- ENTITIES ---")
            for key, ent in self.entities.items():
                attrs_str = ", ".join(_neutralize(f"{k}={v}") for k, v in ent["attributes"].items())
                parts.append(f"[{_neutralize(ent['entity_type']).upper()}] {_neutralize(ent['entity_id'])}: {attrs_str}")
            parts.append("")

        # 2. Multimodal Descriptors (if any). The <|audio|> / <|vision|> tags are the builder's own.
        if self.multimodal_items:
            parts.append("--- MULTIMODAL SENSORS ---")
            for item in self.multimodal_items:
                if item["type"] == "audio":
                    dur_str = f" ({item['duration_sec']}s)" if item["duration_sec"] else ""
                    events = ", ".join(_neutralize(e) for e in item["detected_events"])
                    events_str = f" [Events: {events}]" if item["detected_events"] else ""
                    parts.append(f"<|audio|> Audio: {_neutralize(item['summary'])}{dur_str}{events_str}")
                elif item["type"] == "vision":
                    res_str = f" ({item['resolution'][0]}x{item['resolution'][1]})" if item["resolution"] else ""
                    objects = ", ".join(_neutralize(o) for o in item["objects_detected"])
                    objs_str = f" [Objects: {objects}]" if item["objects_detected"] else ""
                    parts.append(f"<|vision|> Vision: {_neutralize(item['summary'])}{res_str}{objs_str}")
            parts.append("")

        # 3. Content Sections
        for sec in self.sections:
            parts.append(f"--- {sec['name'].upper()} ---")
            parts.append(sec["content"])
            parts.append("")

        return "\n".join(parts).strip()

    def build(self) -> Dict[str, Any]:
        """Produce structured payload dictionary containing state string and metadata."""
        return {
            "state_text": self.build_state_string(),
            "sections_count": len(self.sections),
            "entities": self.entities,
            "multimodal_items": self.multimodal_items,
        }

    def clear(self) -> "ContextBuilder":
        """Reset the builder."""
        self.sections.clear()
        self.entities.clear()
        self.multimodal_items.clear()
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextBuilder":
        """Convenience factory: initialize context builder directly from dictionary."""
        builder = cls()
        builder.add_json(data, section_name="Primary Context")
        return builder

    @classmethod
    def from_tabular(cls, rows: List[Dict[str, Any]], headers: Optional[List[str]] = None) -> "ContextBuilder":
        """Convenience factory: initialize context builder directly from tabular records."""
        builder = cls()
        builder.add_tabular(rows, headers=headers)
        return builder

    @classmethod
    def from_data(cls, data: Any) -> "ContextBuilder":
        """Convenience factory: initialize context builder from dict, list, JSON string, or raw text."""
        builder = cls()
        if isinstance(data, dict):
            builder.add_json(data, section_name="Primary Context")
        elif isinstance(data, list):
            if data and all(isinstance(r, dict) for r in data):
                builder.add_tabular(data, section_name="Primary Context")
            else:
                builder.add_json(data, section_name="Primary Context")
        elif isinstance(data, str):
            stripped = data.strip()
            if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
                try:
                    parsed = json.loads(stripped)
                    return cls.from_data(parsed)
                except Exception:
                    builder.add_text(data, section_name="Primary Context")
            else:
                builder.add_text(data, section_name="Primary Context")
        else:
            builder.add_text(str(data), section_name="Primary Context")
        return builder
