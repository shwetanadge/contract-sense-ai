import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

GDPR_HEADING = re.compile(r"^Article (\d+)$")
HGB_HEADING = re.compile(r"^§ (\d+)([a-z]?)(?: (.+))?$")

# Lines like "§ 268 Abs. 3 auszuweisen..." are references wrapped onto a new
# line, not real headings.
REFERENCE_START = re.compile(
    r"^(?:A ?bs(?:atz)?\b|A ?bs\.|Satz\b|Nr\b|Nummer\b|und\b|oder\b|sowie\b|bis\b|gilt\b|ist\b|sind\b)",
    re.I,
)

def _split_sections(text, source_name):
    """Cut the text into (label, body) sections, one per Article or §."""
    lines = text.split("\n")
    sections = []
    label = ""
    body = []
    last_key = (0, "")   # last HGB section number seen, to ignore references

    for idx, line in enumerate(lines):
        new_label = None

        if source_name == "GDPR":
            m = GDPR_HEADING.match(line)
            if m:
                new_label = f"Article {m.group(1)}"
                if idx + 1 < len(lines):
                    new_label += f" - {lines[idx + 1]}"

        elif source_name == "HGB":
            m = HGB_HEADING.match(line)
            if m:
                key = (int(m.group(1)), m.group(2))
                rest = m.group(3) or ""
                is_reference = bool(REFERENCE_START.match(rest)) or rest.endswith(",")
                if not is_reference and last_key < key and key[0] - last_key[0] <= 100:
                    last_key = key
                    new_label = f"§ {m.group(1)}{m.group(2)}"
                    if m.group(3):
                        new_label += f" {m.group(3)}"

        if new_label:
            if body:
                sections.append((label, "\n".join(body)))
            label = new_label
            body = [line]
        else:
            body.append(line)

    if body:
        sections.append((label, "\n".join(body)))
    return sections


def chunk_text(text, source_name):
    """
    Splits text by Article / § first, then into overlapping chunks.
    Every chunk starts with a header such as
    'GDPR, Article 37 - Designation of the data protection officer'
    so it keeps the context of the article it belongs to.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    result = []
    chunk_id = 0
    for label, body in _split_sections(text, source_name):
        header = f"{source_name}, {label}" if label else source_name
        for piece in splitter.split_text(body):
            result.append({
                "text": f"{header}\n{piece}",
                "metadata": {
                    "source": source_name,
                    "chunk_id": chunk_id,
                    "section": label,
                },
            })
            chunk_id += 1
    return result