
import streamlit as st
import pandas as pd
from vector_store import (
    index_document_chunks,
    search_similar_chunks,
    get_vector_count
)
from retrieval import hybrid_search

from document_store import (
    initialise_database,
    save_document,
    list_documents,
    get_document_chunks,
    get_document_count
)

from document_ingestion import ingest_multiple_pdfs

initialise_database()

st.set_page_config(
    page_title="Enterprise Banking RAG",
    page_icon="🏦",
    layout="wide"
)

st.title("Enterprise Banking RAG Assistant")
st.caption("Milestone 1 — Multi-document ingestion")

uploaded_files = st.file_uploader(
    "Upload banking and financial PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Ingest Documents", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Processing documents..."):
            results = ingest_multiple_pdfs(uploaded_files)

        successful_documents = [
            result["document"]
            for result in results
            if result["success"]
        ]

        st.session_state["ingested_documents"] = successful_documents

        for result in results:
            if result["success"]:
                document = result["document"]
                save_document(document)
                indexed_count = index_document_chunks(
                    document["chunks"]
                )

                st.success(
                    f"Successfully indexed {indexed_count} chunks in ChromaDB."
                )
                st.metric(
                    "Total Indexed Chunks",
                    get_vector_count()
                )
                st.success(
                    f"Processed {document['document_name']} "
                    f"— {document['page_count']} pages, "
                    f"{document['chunk_count']} chunks"
                )
            else:
                st.error(
                    f"{result['file_name']}: {result['error']}"
                )


documents = list_documents()

st.subheader("Enterprise Document Catalogue")

st.metric(
    "Total Active Documents",
    get_document_count()
)

if documents:
    st.dataframe(
        pd.DataFrame(documents)[[
            "document_name",
            "page_count",
            "chunk_count",
            "file_size_bytes",
            "ingested_at",
            "status"
        ]],
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No documents have been ingested yet.")

# Stored Document Chunks — outside the if/else block
st.subheader("Stored Document Chunks")

for document in documents:
    with st.expander(document["document_name"]):

        chunks = get_document_chunks(
            document["document_id"]
        )

        st.write(f"Stored chunks: {len(chunks)}")

        for chunk in chunks[:3]:
            st.caption(
                f"Page {chunk['page_number']} | "
                f"Chunk {chunk['chunk_index']}"
            )

            st.write(chunk["text"])
            st.divider()

st.subheader("Semantic Document Search")

question = st.text_input(
    "Ask a question about your banking documents"
)

if st.button("Search Documents"):
    if question.strip():
        with st.spinner("Searching ChromaDB..."):
            results = search_similar_chunks(
                question=question,
                top_k=5
            )

        if results:
            for result in results:
                metadata = result["metadata"]

                st.markdown(
                    f"**{metadata['document_name']}** "
                    f"| Page {metadata['page_number']}"
                )

                st.write(result["text"])

                st.caption(
                    f"Cosine distance: {result['distance']:.4f}"
                )
                st.divider()
        else:
            st.info("No matching document chunks found.")
    else:
        st.warning("Please enter a search question.")


st.subheader("Enterprise Hybrid Search")

hybrid_question = st.text_input(
    "Ask a question using keyword + semantic retrieval",
    key="hybrid_question"
)

if st.button("Run Hybrid Search"):

    all_chunks = []

    for document in list_documents():
        chunks = get_document_chunks(
            document["document_id"]
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        st.warning("No document chunks are available.")
    elif not hybrid_question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Running hybrid retrieval..."):

            results = hybrid_search(
                question=hybrid_question,
                chunks=all_chunks,
                top_k=5,
                candidate_k=10
            )

        if results:
            for result in results:
                st.markdown(
                    f"**{result['document_name']}** "
                    f"| Page {result['page_number']}"
                )

                st.write(result["text"])

                st.caption(
                    f"RRF score: {result['rrf_score']:.5f} "
                    f"| Methods: {', '.join(result['methods'])}"
                )

                st.divider()
        else:
            st.info("No relevant passages found.")
