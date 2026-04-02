import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        preview = self.text[:80].replace("\n", " ")
        return f"Chunk(doc_type={self.metadata.get('doc_type')}, source={self.metadata.get('source_file')}, text='{preview}...')"


RETRO_SECTION_PATTERN = re.compile(
    r"(###\s+(?:What went well|What didn't go well|Action items))", re.IGNORECASE
)

PRD_SECTION_PATTERN = re.compile(
    r"((?:^TL;DR|^Problem Definition|^Users & Use Cases|^Opportunity|^JTDBs|"
    r"^Requirements and Product Definition|^User Experience|^Success Metrics))",
    re.MULTILINE | re.IGNORECASE,
)


def _split_by_pattern(text: str, pattern: re.Pattern) -> list[str]:
    parts = pattern.split(text)
    sections = []
    if parts[0].strip():
        sections.append(parts[0].strip())
    i = 1
    while i < len(parts) - 1:
        section = (parts[i] + "\n" + parts[i + 1]).strip()
        if section:
            sections.append(section)
        i += 2
    if i < len(parts) and parts[i].strip():
        sections.append(parts[i].strip())
    return [s for s in sections if s]


def _split_by_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _approx_tokens(text: str) -> int:
    return len(text) // 4


def _merge_into_chunks(segments: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    current_parts: list[str] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = _approx_tokens(seg)
        if current_tokens + seg_tokens > chunk_size and current_parts:
            chunks.append("\n\n".join(current_parts))
            while current_parts and current_tokens > overlap:
                removed = current_parts.pop(0)
                current_tokens -= _approx_tokens(removed)
        current_parts.append(seg)
        current_tokens += seg_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def chunk_document(body: str, metadata: dict, chunk_size: int = 400, overlap: int = 80) -> list[Chunk]:
    doc_type = metadata.get("doc_type", "unknown")

    if doc_type == "retro":
        sections = _split_by_pattern(body, RETRO_SECTION_PATTERN)
        raw_chunks = sections if sections else [body]
    elif doc_type == "prd":
        sections = _split_by_pattern(body, PRD_SECTION_PATTERN)
        raw_chunks = _merge_into_chunks(sections or [body], chunk_size, overlap)
    else:
        paragraphs = _split_by_paragraphs(body)
        raw_chunks = _merge_into_chunks(paragraphs or [body], chunk_size, overlap)

    chunks = []
    for i, text in enumerate(raw_chunks):
        if not text.strip():
            continue
        chunks.append(Chunk(
            text=text.strip(),
            metadata={**metadata, "chunk_index": i, "chunk_count": len(raw_chunks)},
        ))

    return chunks
