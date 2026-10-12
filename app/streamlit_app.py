"""Interactive Streamlit application for FEVERScope.

Follows Section 7 (Phase 8):
- Input custom claim.
- Side-by-side Top-10 ranking comparisons across TF-IDF, BM25, Dense, and RRF.
- Gold evidence highlighting for known evaluation claims.
- Metrics & PR curves loaded directly from results/ (never hardcoded numbers).
"""

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="FEVERScope Evidence Retrieval", layout="wide")
    st.title("FEVERScope: Evidence Retrieval Evaluation")
    st.markdown(
        """
        Comparing **TF-IDF**, **BM25**, **Dense (Sentence-Transformers)**, and **RRF Hybrid**
        retrieval across the FEVER Wikipedia sentence corpus.
        """
    )
    st.info("Application interface stub. Full interaction and evaluation views activate in Phase 8.")


if __name__ == "__main__":
    main()
