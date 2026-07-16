# core/compress.py
# Prompt compression — local-to-local only.
#
# This stage used to run in front of everything, including the escalation path,
# on the theory that cheap local tokens buy expensive API tokens. That theory was
# wrong in two ways, and the stage was moved behind the router because of it:
#
#   1. It inverts the quality gradient. Letting the weakest model in the system
#      decide what the strongest models are allowed to see is backwards. A
#      nuanced question, aggressively "compressed" by a 8B local model, reaches
#      the council already damaged — and no amount of frontier reasoning recovers
#      detail that was deleted before it arrived.
#   2. The economics never supported it. Input tokens are the cheap direction.
#      Trading answer quality for a fractional saving on the cheap side of the
#      ledger is a bad trade at any volume.
#
# So: never compress anything bound for a frontier model. Compression is now only
# applied local-to-local, and only when the prompt is big enough that context
# pressure genuinely justifies the lossiness.

from core.clients import ollama_client, LOCAL_MODEL, TEMP_DETERMINISTIC
from core.helpers import estimate_tokens

# Below this, there is no context pressure to relieve and compression is pure
# downside — a lossy rewrite of a prompt that already fit comfortably.
COMPRESS_MIN_TOKENS = 200

COMPRESSION_SYSTEM = """You are a compression agent. Rewrite the given prompt in the most token-efficient form possible while preserving complete meaning.

Rules:
- Remove filler words, pleasantries, and redundant phrasing
- Remove phrases like 'Could you please', 'I would like to know', 'can you tell me'
- Collapse verbose phrasing into direct questions
- Preserve all key terms, named entities, and specific requirements
- Never lose meaning — compression must be lossless in terms of intent

Return only the compressed prompt as a plain string. No JSON, no explanation, no preamble."""


def compress_prompt(prompt):
    """Compress a verbose prompt via the local LLM.

    Returns a dict with the compressed text and before/after token estimates
    so the pipeline can report how much was saved.
    """
    result = ollama_client.chat(
        model=LOCAL_MODEL,
        messages=[
            {"role": "system", "content": COMPRESSION_SYSTEM},
            {"role": "user", "content": f"Compress this prompt: {prompt}"},
        ],
        options={"temperature": TEMP_DETERMINISTIC},
    )

    compressed = result["message"]["content"].strip()

    # Strip any accidental quotes the model might wrap around the output
    if compressed.startswith('"') and compressed.endswith('"'):
        compressed = compressed[1:-1].strip()

    original_tokens = estimate_tokens(prompt)
    compressed_tokens = estimate_tokens(compressed)
    reduction = round((1 - compressed_tokens / original_tokens) * 100) if original_tokens > 0 else 0

    return {
        "compressed": compressed,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "reduction_percent": reduction,
    }


def maybe_compress(prompt):
    """Compress only if the prompt is large enough for it to be worth the loss.

    Returns the same dict as compress_prompt(), or None when the prompt is small
    enough that compressing it would trade meaning for nothing.

    Callers must only reach this on the local path. Anything bound for a frontier
    model gets the original prompt, verbatim — see the module header.
    """
    if estimate_tokens(prompt) < COMPRESS_MIN_TOKENS:
        return None
    return compress_prompt(prompt)
