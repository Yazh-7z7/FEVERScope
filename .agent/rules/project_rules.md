# Agent Operating Protocol (Non-Negotiable)

1. **Work on exactly one component per task, within the current phase only.** Never start the next phase without explicit human approval.
2. **Planning first.** For every task, produce an implementation plan artifact and wait for human review before writing code.
3. **Never fabricate** data, example outputs, or metrics. Every number reported anywhere (README, app, report) must come from a file in `results/` that was produced by code in this repo.
4. **Never open, `cat`, or print whole files under `data/`** (multi-GB). Use streaming iteration, `head -n`, or sampling.
5. **The human runs the large downloads** (Section 5, Step 3). Do not download `wiki-pages.zip` yourself.
6. **Learning mode:** after each task, append a 5-line plain-English explanation: what was built, why this design, what to look at in the output, one thing that could go wrong, one question the human should be able to answer.
7. **Tests are part of the task.** Each module ships with pytest tests using tiny, hand-verifiable fixtures. Report the test output.
8. **Small diffs.** Prefer small, reviewable changes. Do not refactor unrelated code.
9. **Determinism.** Fixed seeds from config. Use `hashlib` (not Python's built-in `hash()`, which is salted per process) wherever a stable hash is needed.
10. **No hidden constants.** All paths, seeds, sample sizes, and hyperparameters come from `configs/default.yaml`.
