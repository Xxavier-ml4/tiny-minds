# Security policy

## Supported versions

Pre-1.0: only the `main` branch is supported. There are no tagged releases
yet (see `CHANGELOG.md`).

## Reporting a vulnerability

Do not open a public issue for a security vulnerability. Once this
repository has a public home, report privately to the maintainers listed
there. Include: what you found, how to reproduce it, and the potential
impact. Expect an acknowledgment within a few days.

## Threat model (current)

This is a local-first runtime. The main risks this project actively designs
against, per `docs/architecture/tinymind-design.md` §12:

1. **Malicious or malformed model files.** `tinymind/runtime/format.py`
   bounds-checks every offset, size, and count read from a `.tm` file
   against the actual file size before touching tensor data, and raises a
   specific `ModelFormatError` subclass rather than trusting the header. See
   `tests/test_format.py` for the malformed-file fixtures this is tested
   against (truncated file, out-of-bounds tensor offset, bad magic, unknown
   format version).
2. **Malicious tool arguments / prompt injection.** Tool execution
   (`tinymind/tools/executor.py`) only ever invokes a function object
   already present in the `ToolRegistry` — there is no code path from model
   output, tool arguments, or user input to `eval`, `exec`, or
   `subprocess(..., shell=True)` anywhere in this codebase. Destructive or
   sensitive tools (`tinymind/tools/permissions.py`) require an explicit
   confirmation flag from the calling application; the runtime never
   supplies that confirmation on the model's behalf.
3. **Unintended network exposure.** The `tinymind serve` HTTP command (once
   implemented — see `STATUS.md`) is specified to bind `127.0.0.1` unless
   `--host` is passed explicitly, the same default Needle's own playground
   server uses and for the same reason.
4. **Arbitrary code execution via untrusted schemas.** The JSON-Schema
   validator in `tinymind/runtime/constraints/json_schema.py` is a plain
   recursive validator with no `eval` of pattern strings beyond Python's
   `re` module against a bounded input; a schema itself cannot cause code
   execution.

## Out of scope for this delivery

Native-engine memory safety (buffer overflows, integer overflows in a
compiled `.so`) is out of scope until `native/` has a real implementation —
see `STATUS.md`. The interface stubs currently in `native/src/` do not parse
untrusted input.
