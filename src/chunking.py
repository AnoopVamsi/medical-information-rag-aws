import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGES_PATH = PROJECT_ROOT / "data" / "cache" / "pages.json"
CHUNKS_PATH = PROJECT_ROOT / "data" / "cache" / "chunks.json"

CHUNK_WORDS = 120
OVERLAP_WORDS = 20

# Headings verified for our first article.
HEADINGS = {
    "What Is Diabetes?",
    "What are the different types of diabetes?",
    "Type 1 diabetes",
    "Type 2 diabetes",
    "Gestational diabetes",
    "Prediabetes",
    "Other types of diabetes",
    "How common are diabetes and prediabetes?",
    "What other health problems can people with diabetes develop?",
}


def build_chunks(pages, chunk_words=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    if chunk_words <= 0 or not 0 <= overlap < chunk_words:
        raise ValueError("Overlap must be smaller than a positive chunk size.")

    chunks = []
    current_sections = {}
    ended_documents = set()

    for page in pages:
        document_id = page["document_id"]

        if document_id in ended_documents:
            continue

        section = current_sections.get(document_id, page["title"])
        lines = []
        segments = []

        # Join a heading that wraps across two lines in this PDF.
        text = page["text"].replace(
            "What other health problems can people with diabetes\ndevelop?",
            "What other health problems can people with diabetes develop?",
        )

        for line in text.splitlines():
            line = line.strip()

            if line == "References":
                ended_documents.add(document_id)
                break

            if line in HEADINGS:
                if lines:
                    segments.append((section, " ".join(lines)))
                    lines = []

                section = line
            elif line:
                lines.append(line)

        if lines:
            segments.append((section, " ".join(lines)))

        current_sections[document_id] = section
        page_chunk_number = 0

        for section_name, segment_text in segments:
            words = segment_text.split()
            start = 0

            while start < len(words):
                end = min(start + chunk_words, len(words))
                page_chunk_number += 1

                chunk = {
                    key: value
                    for key, value in page.items()
                    if key != "text"
                }
                chunk.update(
                    {
                        "chunk_id": (
                            f"{document_id}"
                            f"_p{page['page_number']}"
                            f"_c{page_chunk_number}"
                        ),
                        "section": section_name,
                        "text": " ".join(words[start:end]),
                        "word_count": end - start,
                    }
                )
                chunks.append(chunk)

                if end == len(words):
                    break

                start = end - overlap

    return chunks


def main():
    try:
        pages = json.loads(PAGES_PATH.read_text(encoding="utf-8"))
        chunks = build_chunks(pages)

        if not chunks:
            raise ValueError("No chunks created.")

        CHUNKS_PATH.write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Chunking failed: {error}") from error

    print(f"Saved {len(chunks)} chunks.")

    for chunk in chunks:
        print(
            f"{chunk['chunk_id']} | "
            f"{chunk['section']} | "
            f"{chunk['word_count']} words"
        )


if __name__ == "__main__":
    main()