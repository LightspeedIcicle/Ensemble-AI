# core/compress.py
# Stage 1 — Prompt compression.
# Uses the local LLM to strip verbose prompts down to token-efficient shorthand
# before they ever reach the paid external models. Cheap tokens buy expensive ones.

from core.clients import ollama_client, LOCAL_MODEL
from core.helpers import estimate_tokens

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
