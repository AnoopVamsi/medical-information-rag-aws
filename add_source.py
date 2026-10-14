"""Register the second NIDDK document without replacing existing entries."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "data/sources.json"

document = {
    "document_id": "niddk_symptoms_causes",
    "filename": "niddk_symptoms_causes.pdf",
    "title": "Symptoms & Causes of Diabetes",
    "publisher": (
        "National Institute of Diabetes and Digestive "
        "and Kidney Diseases (NIDDK), NIH"
    ),
    "source_url": (
        "https://www.niddk.nih.gov/health-information/"
        "diabetes/overview/symptoms-causes"
    ),
    "last_reviewed": "2024-10",
    "last_reviewed_precision": "month",
    "snapshot_date": "2026-10-05",
    "format": "browser_printed_pdf",
    "page_reference_type": "local_pdf_snapshot",
    "reuse_policy_url": "https://www.niddk.nih.gov/copyright",
    "reuse_notes": (
        "NIDDK permits reuse of most content with attribution, "
        "subject to exceptions. Extract article text only; "
        "do not reuse logos or third-party graphics or imply endorsement."
    ),
}

data = json.loads(SOURCES.read_text(encoding="utf-8"))
documents = data["documents"]

if any(
    item["document_id"] == document["document_id"]
    for item in documents
):
    print("Document already registered; no changes made.")
else:
    documents.append(document)
    SOURCES.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Registered documents: {len(documents)}")