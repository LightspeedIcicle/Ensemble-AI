# core/chunking.py
# Splitting text into embeddable chunks.
#
# This is pure logic and lives here rather than in memory.py so it can be tested
# without importing chromadb, sentence-transformers, and pypdf. It was previously
# inside memory.py, where exercising a string-splitting function meant loading a
# vector database — so it was never exercised at all.

# all-MiniLM-L6-v2 silently truncates its input at ~256 tokens. At roughly four
# characters per token that ceiling lands at ~1000 characters, which is exactly
# where CHUNK_SIZE sits. Raising CHUNK_SIZE does not get you bigger memories --
# it gets you memories whose tails are dropped before they are ever embedded,
# with no error, and retrieval quality degrades silently. Do not raise this
# without switching to an embedder with a longer context.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ordered strongest-to-weakest. A chunk should end at a paragraph break if one is
# available, and only fall back to a mid-sentence cut when nothing better exists.
_BREAKS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks that end on natural boundaries.

    Hard-slicing every `chunk_size` characters cuts sentences, formulas, and code
    blocks in half, so retrieval returns fragments that start mid-thought. This
    walks to the nearest real boundary instead, and overlaps consecutive chunks so
    a fact spanning a boundary survives in at least one of them.
    """
    text = (text or "").strip()
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            # Only look for a boundary in the last part of the window, so we
            # don't emit a tiny chunk just because an early newline exists.
            floor = start + int(chunk_size * 0.6)
            for sep in _BREAKS:
                idx = text.rfind(sep, floor, end)
                if idx != -1:
                    end = idx + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        # max() guards against a pathological chunk shorter than the overlap,
        # which would otherwise never advance.
        start = max(end - overlap, start + 1)

    return chunks
