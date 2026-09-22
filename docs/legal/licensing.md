# Licensing and provenance

This document explains, plainly, where TinyMind's ideas came from and what
was and was not reused from Needle 2.

## Short version

- TinyMind is licensed Apache-2.0, same as Needle.
- Needle's source was studied in full before any TinyMind code was written
  (`docs/architecture/needle-analysis.md`).
- No Needle source files, class names, function signatures, tests, prompts,
  model weights, branding, or artwork appear in this repository.
- Where a TinyMind design decision was directly inspired by a specific
  Needle mechanism, that is stated explicitly, next to the decision, in
  `docs/architecture/tinymind-design.md` — not left for a reader to guess.
- TinyMind does not use "Needle," "Cactus," "SAN" (Simple Attention
  Network), "Engram," ".cact," or any other Needle/Cactus-Compute name in
  its own public API, file formats, or CLI.

## Why this matters

Apache-2.0 is a permissive license: it would allow TinyMind to copy Needle's
source directly, rename things, and ship it. That is explicitly **not** what
this project set out to do — the engineering brief this repository was built
from states the goal as "design a better general-purpose tiny AI runtime and
training stack inspired by what Needle got right, while fixing its
limitations," not "make another Needle." A mechanical rename would also be a
worse outcome on the merits: Needle's own design has real, documented
limitations (see needle-analysis.md §23), and a fork inherits them by
construction.

## What "independent implementation" meant in practice

For every subsystem, the working method was: read the real Needle
implementation, write down what it does and why, identify what's worth
keeping as a *concept*, then write TinyMind's version from that
understanding rather than by editing a copy of Needle's file. The
architecture analysis and design documents in `docs/architecture/` are the
paper trail for that process — they were written *before* the corresponding
TinyMind code, not after, so the reasoning is not reconstructed after the
fact.

Concretely, every TinyMind module differs from its Needle counterpart in at
least one of: language-level structure (different class hierarchy, not a
renamed copy), scope (TinyMind's version does more, less, or something
different — see `tinymind-design.md` per subsystem), or is for a concept
Needle does not have exposed in Python at all (e.g., TinyMind's constraint
engine is Python and testable; Needle's is a closed native binary).

## If Apache-2.0 code is ever directly incorporated

This repository does not currently contain any file copied from Needle. If a
future change does incorporate Needle (or any other Apache-2.0) source
directly, Apache License 2.0 §4 requires, at minimum:

1. The recipient must be given a copy of the License (already satisfied —
   `LICENSE` at the repository root).
2. Modified files must carry prominent notices stating they were changed.
3. The original copyright, patent, trademark, and attribution notices from
   the source must be retained in any Derivative Works, in at least one of
   the customary places (a `NOTICE` file, file headers, or documentation).
4. `NOTICE` must continue to include the attribution notices above, if the
   original work included a `NOTICE` file.

None of this is a formality to route around — it is the actual legal
mechanism by which reuse under Apache-2.0 stays honest about where code came
from, which is the same goal this document and `NOTICE` serve today even
though nothing has been directly copied yet.

## Trademarks

Apache-2.0 explicitly does not grant trademark rights (§4, penultimate
paragraph, and §6). "Cactus," "Needle," and related marks, if any, belong to
their owner. TinyMind's own name is a working project name (see
`README.md`) chosen specifically so it can change without any of this
document needing to change.
