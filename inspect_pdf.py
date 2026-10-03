from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "niddk_what_is_diabetes.pdf"
)


def main():
    if not PDF_PATH.is_file():
        raise SystemExit(f"PDF not found: {PDF_PATH}")

    try:
        with pymupdf.open(PDF_PATH) as document:
            if document.needs_pass:
                raise SystemExit("The PDF requires a password.")

            print(f"File: {PDF_PATH.name}")
            print(f"Pages: {len(document)}")

            total_characters = 0

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text", sort=True).strip()
                total_characters += len(text)

                print(
                    f"\nPage {page_number}: "
                    f"{len(text)} text characters"
                )

                if text:
                    print("Preview:")
                    print(text[:400])
                else:
                    print("No text extracted from this page.")

            print(f"\nTotal text characters: {total_characters}")

            if total_characters == 0:
                raise SystemExit(
                    "No readable text found. "
                    "Check whether the PDF contains scanned images."
                )

    except (pymupdf.FileDataError, RuntimeError) as error:
        raise SystemExit(f"Could not read the PDF: {error}") from error


if __name__ == "__main__":
    main()