# datasets/

Per the engineering brief section 46, training data flows through four
stages, each its own subdirectory:

- `seed/` — the starting corpus, used as **reference material only**, not
  trained on directly. Empty in this delivery: no seed dataset ships here
  (see `STATUS.md`) — `tinymind.data.validate_file` is ready to check
  whatever lands here against the format in the brief's section 22 the
  moment there is something to check.
- `validated/` — seed examples that passed `tinymind.data.validate_file`
  and `tinymind.data.deduplicate`. Empty until `seed/` is populated.
- `generated/` — synthetic examples from a distillation teacher
  (`tinymind.distillation`, not implemented — needs a configured teacher
  endpoint, see that package's docstring). Empty.
- `heldout/` — examples deliberately excluded from training for
  evaluation. Empty until there's a training run to hold examples out of.

None of these directories should be silently treated as "the project has
no data pipeline" — the pipeline (`tinymind.data`, `tinymind.training.
dataset`) is real and tested against synthetic examples in `tests/
test_data.py` and `tests/test_config_and_deferred_subsystems.py`; what's
missing is the data itself, which is a content question, not a code
question, and the two are deliberately kept separable.
