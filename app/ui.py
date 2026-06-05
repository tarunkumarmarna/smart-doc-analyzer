"""
ui.py — Streamlit frontend for Smart Document Analyzer
Two tabs:
  Tab 1 — Analyze: upload PDF or paste text → classification + entities + summary
  Tab 2 — Ask: semantic search over processed document

Communicates with FastAPI backend via HTTP requests.
WHY separate UI from API: UI can be swapped (React, Gradio etc) without touching ML code.
"""

import streamlit as st    # Streamlit for interactive web UI
import requests           # HTTP client to call FastAPI backend
import plotly.express as px  # for confidence score visualization
import json               # for handling JSON data

# FastAPI backend URL
# Change to Render URL after deployment
API_URL = "http://localhost:8000"

# --- Page Config ---
st.set_page_config(
    page_title="Smart Document Analyzer",
    page_icon="📄",
    layout="wide"   # wide layout uses full browser width
)

# --- Title ---
st.title("📄 Smart Document Analyzer")
st.caption("Upload any PDF or paste text. Ask questions in plain English, get answers from the document.")

# --- Session State ---
# WHY session_state: Streamlit re-runs entire script on every interaction.
# session_state persists data across reruns within same browser session.
if "doc_data" not in st.session_state:
    st.session_state.doc_data = None  # stores process_document() response

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Analyze Document", "🔍 Ask Questions"])


# ===================== TAB 1 — ANALYZE =====================
with tab1:
    st.subheader("Upload or paste your document")

    # input method selector
    input_method = st.radio(
        "Input method",
        ["Upload PDF", "Paste Text"],
        horizontal=True   # display options side by side
    )

    doc_ready = False  # flag: True when document is uploaded and ready to process

    if input_method == "Upload PDF":
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Digital PDFs only. Scanned PDFs not supported in V1."
        )
        if uploaded_file:
            doc_ready = True
            st.success(f"✅ Loaded: {uploaded_file.name} ({uploaded_file.size // 1024} KB)")

    else:  # Paste Text
        pasted_text = st.text_area(
            "Paste your text here",
            height=200,
            placeholder="Paste any document text here..."
        )
        if pasted_text.strip():
            doc_ready = True

    # Analyze button — only shown when document is ready
    if doc_ready:
        if st.button("🔍 Analyze Document", type="primary"):
            with st.spinner("Analyzing document... this may take 20-30 seconds on first run"):
                try:
                    if input_method == "Upload PDF":
                        # send PDF as multipart form data to /process
                        response = requests.post(
                            f"{API_URL}/process",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        )
                    else:
                        # send plain text to /process-text
                        response = requests.post(
                            f"{API_URL}/process-text",
                            json={"text": pasted_text}
                        )

                    if response.status_code == 200:
                        # store result in session_state so Tab 2 can access it
                        st.session_state.doc_data = response.json()
                        st.success("✅ Analysis complete!")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Make sure FastAPI is running on port 8000.")

    # --- Display Results ---
    if st.session_state.doc_data:
        data = st.session_state.doc_data

        # layout: 3 columns for key metrics at top
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Document Type", data["classification"]["label"])
        with col2:
            confidence_pct = f"{data['classification']['confidence'] * 100:.1f}%"
            st.metric("Classification Confidence", confidence_pct)
        with col3:
            st.metric("Chunks Extracted", data["chunk_count"])

        st.divider()

        # --- Classification confidence chart ---
        st.subheader("📊 Classification Scores")
        all_scores = data["classification"]["all_scores"]

        # create bar chart using plotly
        # px.bar handles list of labels and scores directly
        fig = px.bar(
            x=list(all_scores.keys()),
            y=list(all_scores.values()),
            labels={"x": "Category", "y": "Confidence Score"},
            color=list(all_scores.values()),
            color_continuous_scale="Blues"  # blue gradient by score
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --- Entities ---
        st.subheader("🏷️ Named Entities")
        entities = data["entities"]

        if entities:
            # display each entity type in its own column
            entity_cols = st.columns(len(entities))
            entity_labels = {"PER": "👤 People", "ORG": "🏢 Organizations",
                           "LOC": "📍 Locations", "MISC": "🔖 Miscellaneous"}

            for i, (etype, words) in enumerate(entities.items()):
                with entity_cols[i % len(entity_cols)]:
                    label = entity_labels.get(etype, etype)
                    st.write(f"**{label}**")
                    for word in words:
                        st.write(f"• {word}")
        else:
            st.info("No named entities detected.")

        st.divider()

        # --- Summary ---
        st.subheader("📝 Document Summary")
        st.write(data["summary"])

        st.divider()

        # --- Raw text preview ---
        with st.expander("📄 View extracted text (first 1000 chars)"):
            st.text(data["full_text"][:1000] + "..." if len(data["full_text"]) > 1000 else data["full_text"])


# ===================== TAB 2 — SEARCH =====================
with tab2:
    st.subheader("Ask questions about your document")

    # check if document has been processed
    if not st.session_state.doc_data:
        st.info("👈 Please analyze a document in the **Analyze Document** tab first.")
    else:
        st.success(f"✅ Document ready — {st.session_state.doc_data['chunk_count']} chunks indexed")

        # search input
        query = st.text_input(
            "Ask a question",
            placeholder="e.g. What were the key findings? What is the revenue growth?",
        )

        if query.strip():
            if st.button("🔍 Search", type="primary"):
                with st.spinner("Searching..."):
                    try:
                        # send query + document chunks + embeddings to /search
                        response = requests.post(
                            f"{API_URL}/search",
                            json={
                                "query": query,
                                "chunks": st.session_state.doc_data["chunks"],
                                "chunk_embeddings": st.session_state.doc_data["chunk_embeddings"]
                            }
                        )

                        if response.status_code == 200:
                            search_data = response.json()
                            results = search_data["results"]

                            st.subheader(f"Top {len(results)} relevant sections")

                            # display each result with similarity score
                            for i, result in enumerate(results):
                                score_pct = f"{result['score'] * 100:.1f}%"

                                # use expander so user can expand/collapse each result
                                with st.expander(f"Result {i+1} — Similarity: {score_pct}"):
                                    st.write(result["chunk"])
                                    st.caption(f"Chunk index: {result['chunk_index']} | Score: {result['score']}")
                        else:
                            st.error(f"Search error: {response.json().get('detail', 'Unknown error')}")

                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend.")
