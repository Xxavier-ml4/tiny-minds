# Contributing to TinyMind

TinyMind is early: Phase 1 (architecture + reference runtime) per
`docs/architecture/tinymind-design.md`. Before opening a PR, check
`STATUS.md` for what's real versus designed-only — it saves everyone a
review cycle.

## Ground rules

1. **No feature is "done" until it has a test.** The project's own working
   style (`docs/architecture/tinymind-design.md` §0) is: real, runnable code
   with tests, or a clearly-labeled interface stub — never something in
   between that looks finished but isn't. See `tests/` for the existing
   pattern (stdlib `unittest`, no dependency required to run it).
2. **Don't claim a phase is complete in `STATUS.md` until it is.** If your
   change moves a subsystem from "interface only" to "implemented," update
   its row in `STATUS.md` in the same PR.
3. **New tools never get `eval`/`exec`/`shell=True` access to model or tool
   output.** See `tinymind/tools/executor.py` and `docs/architecture/
   tinymind-design.md` §12 for why this is a hard rule, not a style
   preference.
4. **Keep modules small.** If a file is doing two unrelated things, split
   it — the brief this project was built from explicitly calls out "no
   giant monolithic classes" as a requirement, not a suggestion.
5. **If you reuse Apache-2.0 code from Needle or elsewhere**, read
   `docs/legal/licensing.md` first — there's a specific, small set of
   notice requirements to follow, not a blanket "the license allows it."

## Local setup

```sh
git clone <this-repo>
cd TinyMind
pip install -e . --break-system-packages   # or use a venv
make test
```

No network access and no GPU are required to build or test anything in this
delivery — that's deliberate (see `docs/architecture/tinymind-design.md`
§0), and a change that silently adds a hard network or GPU dependency to the
core runtime (as opposed to `training`/`distillation`, which will eventually
need one) should explain why in the PR description.

## Reporting issues

Use the issue tracker once this repository has a public home. For security
issues specifically, see `SECURITY.md` — do not open a public issue.
