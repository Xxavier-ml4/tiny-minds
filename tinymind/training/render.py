"""Compatibility re-export: the canonical example format and renderer live in
``tinymind.data.render`` because *inference* needs them too (a model must be
prompted exactly as it was trained) and inference must not depend on the
training package."""
from tinymind.data.render import (  # noqa: F401
    IGNORE_INDEX, ROLES, TEMPLATE_ID, ChatRenderer, ExampleError, NormalizedExample, RenderedExample, Segment,
    canonical_json, normalize_example, tool_call_text)
