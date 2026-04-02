import re
from pathlib import Path
from typing import Optional


DOC_TYPE_MAP = {
    "interviews": "interview",
    "retros": "retro",
    "prds": "prd",
    "tickets": "ticket",
}


def detect_doc_type(filepath: Path) -> str:
    for folder, doc_type in DOC_TYPE_MAP.items():
        if folder in filepath.parts:
            return doc_type
    return "unknown"


def parse_header_block(text: str) -> tuple[dict, str]:
    header_match = re.match(r"^---\n(.*?)\n---\n?(.*)", text, re.DOTALL)
    if not header_match:
        return {}, text

    header_raw, body = header_match.group(1), header_match.group(2)
    metadata = {}
    for line in header_raw.strip().splitlines():
        if ": " in line:
            key, _, value = line.partition(": ")
            metadata[key.strip().lower().replace(" ", "_")] = value.strip()

    return metadata, body.strip()


def parse_interview(text: str, filepath: Path) -> tuple[dict, str]:
    metadata, body = parse_header_block(text)
    metadata["doc_type"] = "interview"
    metadata["source_file"] = filepath.name
    return metadata, body


def parse_retro(text: str, filepath: Path) -> tuple[dict, str]:
    metadata, body = parse_header_block(text)
    metadata["doc_type"] = "retro"
    metadata["source_file"] = filepath.name
    return metadata, body


def parse_prd(text: str, filepath: Path) -> tuple[dict, str]:
    lines = text.strip().splitlines()
    title_line = lines[0].strip() if lines else ""
    title = re.sub(r"^\[PRD\]\s*", "", title_line).strip()
    metadata = {
        "doc_type": "prd",
        "title": title,
        "source_file": filepath.name,
    }
    body = "\n".join(lines[1:]).strip()
    return metadata, body


def parse_ticket(text: str, filepath: Path) -> tuple[dict, str]:
    metadata, body = parse_header_block(text)
    metadata["doc_type"] = "ticket"
    metadata["source_file"] = filepath.name
    return metadata, body


PARSERS = {
    "interview": parse_interview,
    "retro": parse_retro,
    "prd": parse_prd,
    "ticket": parse_ticket,
}


def parse_document(filepath: Path) -> Optional[tuple[dict, str]]:
    doc_type = detect_doc_type(filepath)
    if doc_type == "unknown":
        print(f"[WARN] Could not detect type for {filepath}, skipping.")
        return None

    text = filepath.read_text(encoding="utf-8")
    metadata, body = PARSERS[doc_type](text, filepath)

    if not body.strip():
        print(f"[WARN] Empty body after parsing {filepath.name}, skipping.")
        return None

    return metadata, body
