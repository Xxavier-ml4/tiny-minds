# benchmarks/mobile/ — not yet populated

Per the engineering brief section 26, this directory is reserved for the
mobile benchmark layer. It is empty in this delivery because it needs a
trained TinyMind model (Phase 3+) and, for on-device numbers, a compiled native runtime (Phase 7+) to
produce a real result against — see `STATUS.md` at the repository root.

The reusable *framework* piece that doesn't need a model,
`tinymind.evaluation.suite.AcceptanceSuite`, already exists and is
demonstrated end to end in `benchmarks/tools/desk_suite.py`; suites in
this directory should build on the same framework once there's a model to
run them against, rather than inventing a second harness.
