# FEVER Data Findings

This document records inspection observations from Phase 1, Step 4 (`scripts/inspect_fever.py`).

## Questions to Answer
1. **Fraction of claims needing multi-sentence or multi-page evidence**:
   - Multi-sentence: TBD
   - Multi-page: TBD
2. **Mean gold sentences per claim (flattened, deduplicated)**:
   - SUPPORTS: TBD
   - REFUTES: TBD
3. **Implications for P@10 ceiling**:
   - Because precision at K is bounded by `min(|rel|, K) / K`, claims with only 1–2 gold sentences have an intrinsic P@10 ceiling of 0.10–0.20. Recall@K, MAP, and nDCG will be critical metrics alongside P@K.
4. **Wikipedia parser observations**:
   - Shard count, line formatting, PTB escapes, and NFC normalization notes.
