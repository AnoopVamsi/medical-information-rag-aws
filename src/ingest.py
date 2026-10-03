import json
import re
from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = PROJECT_ROOT / "data" / "sources.json"
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cache" / "pages.json"

PRINT_HEADER = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{4},?\s+\d{1,2}:\d{2}.*$"
)


def clean_page_text(text):
    """Remove print artifacts and closing website information."""
    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("Last Reviewed"):
            break

        if PRINT_HEADER.fullmatch(line):
            continue

        if re.fullmatch(r"\d+\s*/\s*\d+", line):
            continue

        if line.startswith(
            "https://www.niddk.nih.gov/"
            "health-information/diabetes/overview/what-is-diabetes"
        ):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def load_pages():
    """Extract pages and attach their source metadata."""
    if not SOURCES_PATH.is_file():
        raise ValueError(f"Source metadata not found: {SOURCES_PATH}")

    try:
        metadata = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("data/sources.json is not valid JSON.") from error

    if not isinstance(metadata, dict):
        raise ValueError("Source metadata must be a JSON object.")

    sources = metadata.get("documents")

    if not isinstance(sources, list) or not sources:
        raise ValueError("Source metadata must contain a documents list.")

    pages = []
    required_fields = (
        "document_id",
        "filename",
        "title",
        "publisher",
        "source_url",
        "last_reviewed",
        "page_reference_type",
    )

    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each source must be a JSON object.")

        for field in required_fields:
            if not isinstance(source.get(field), str) or not source[field]:
                raise ValueError(f"Source has a missing or invalid {field}.")

        filename = source["filename"]

        if Path(filename).name != filename:
            raise ValueError("Source filename must not contain a folder path.")

        pdf_path = DOCUMENTS_DIR / filename

        if not pdf_path.is_file():
            raise ValueError(f"PDF not found: {pdf_path}")

        document_pages = []

        with pymupdf.open(pdf_path) as document:
            if document.needs_pass:
                raise ValueError(f"PDF requires a password: {filename}")

            for page_number, page in enumerate(document, start=1):
                raw_text = page.get_text("text", sort=True)
                text = clean_page_text(raw_text)

                if not text:
                    print(
                        f"Skipped {filename}, page {page_number}: "
                        "no text remains after cleaning."
                    )
                    continue

                document_pages.append(
                    {
                        **source,
                        "page_number": page_number,
                        "text": text,
                    }
                )

        if not document_pages:
            raise ValueError(f"No readable text found in {filename}.")

        pages.extend(document_pages)

    return pages


def main():
    try:
        pages = load_pages()
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(pages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"Ingestion failed: {error}") from error

    print(f"Saved {len(pages)} pages to {OUTPUT_PATH}")

    for page in pages:
        print(
            f"{page['document_id']} | "
            f"page {page['page_number']} | "
            f"{len(page['text'])} characters"
        )


if __name__ == "__main__":
    main()