"""Local interface for searching published diabetes education passages."""

import streamlit as st

from src.retrieval import retrieve

st.set_page_config(
    page_title="Medical Information Evidence Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("Medical Information Evidence Assistant")
st.write("Explore published diabetes education information and its sources.")

st.warning(
    "For medical education only. This application does not provide "
    "personalized diagnoses, medication doses, or treatment decisions. "
    "Do not enter patient records or personal health information."
)

st.info(
    "Current stage: local evidence search. Answer generation is not "
    "connected, and live Amazon Bedrock integration has not been verified."
)

with st.form("evidence_search"):
    question = st.text_area(
        "Enter a general educational question about diabetes",
        placeholder="What is type 2 diabetes?",
        max_chars=1000,
    )
    top_k = st.selectbox(
        "Number of passages",
        options=[3, 5, 1],
    )
    submitted = st.form_submit_button("Search evidence")

if submitted:
    if not question.strip():
        st.error("Please enter a question.")
    else:
        try:
            with st.spinner("Searching your local documents..."):
                passages = retrieve(question, top_k=top_k)

            if not passages:
                st.info(
                    "The selected documents do not provide enough information"
                )
            else:
                st.subheader("Retrieved evidence")
                st.caption(
                    "These are similarity-ranked passages, not a generated "
                    "answer. A match does not establish that the evidence "
                    "is sufficient to answer your question."
                )

                for rank, passage in enumerate(passages, start=1):
                    metadata = passage["metadata"]
                    label = (
                        f"{rank}. {metadata['section']} — "
                        f"local PDF page {metadata['page_number']}"
                    )

                    with st.expander(label, expanded=(rank == 1)):
                        # Render the passage as text, not executable markup.
                        st.text(passage["text"])

                        st.write("**Document:**", metadata["title"])
                        st.write("**Publisher:**", metadata["publisher"])
                        st.write(
                            "**Last reviewed:**",
                            metadata.get("last_reviewed", "Not recorded"),
                        )
                        st.link_button(
                            "Open publisher source",
                            metadata["source_url"],
                        )
                        st.caption(
                            f"Chunk ID: {passage['chunk_id']} | "
                            f"Cosine distance: {passage['distance']:.4f}"
                        )
                        st.caption(
                            "Page numbers refer to your saved PDF snapshot. "
                            "The source link opens the publisher's web page."
                        )

        except (OSError, ValueError, KeyError, RuntimeError):
            st.error(
                "Evidence search failed. Check that ingestion, chunking, "
                "embeddings, and indexing completed successfully. "
                "Run the same question with python -m src.retrieval "
                "in your terminal to inspect the error."
            )