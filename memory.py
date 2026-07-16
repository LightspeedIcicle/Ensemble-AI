import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# ── Embedding / chunking configuration ────────────────────────────────────────
# memory.py is a standalone RAG tool, not part of the core/ request path, so it
# owns its own model config rather than importing core.clients (which would drag
# in the paid API clients and require their keys just to index a folder).
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 silently truncates its input at ~256 tokens. At roughly four
# characters per token that ceiling lands at ~1000 characters, which is exactly
# where CHUNK_SIZE sits. Raising CHUNK_SIZE does not get you bigger memories --
# it gets you memories whose tails are dropped before they are ever embedded,
# with no error, and retrieval quality degrades silently. Do not raise this
# without switching to an embedder with a longer context.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 1. Setup the Local Database
# This saves the database to a folder named "chroma_db"
client = chromadb.PersistentClient(path="./chroma_db")

# 2. Setup the Embedding Function (The Translator)
# We use a free, local model to turn text into math. No API calls needed.
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL_NAME
)

# 3. Get or Create the Collection (The Bookshelf)
collection = client.get_or_create_collection(
    name="enscio_knowledge",
    embedding_function=sentence_transformer_ef
)

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


def ingest_knowledge():
    """Reads all files in 'knowledge/' and memorizes them."""
    print("--- MEMORY: Scanning Knowledge Folder ---")
    folder = "knowledge"
    
    # Clear old memory to avoid duplicates (Simple approach)
    # In production, we would check IDs. For now, we rebuild.
    existing_count = collection.count()
    if existing_count > 0:
        print(f"Memory: Found {existing_count} existing records.")
        # Optional: collection.delete(where={}) # Uncomment to wipe and rebuild
    
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    
    for file in files:
        file_path = os.path.join(folder, file)
        text = ""
        
        # Reader Logic
        if file.endswith(".pdf"):
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            except Exception as e:
                print(f"Error reading PDF {file}: {e}")
                continue
        elif file.endswith(".txt") or file.endswith(".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        
        if not text:
            continue
            
        # Chunking (Breaking big text into small memories)
        chunks = split_text(text)
        if not chunks:
            continue

        ids = [f"{file}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file} for _ in range(len(chunks))]

        try:
            # Overlapping chunks yield a different fragment count than the old
            # fixed-slice scheme did, so upsert alone would leave orphaned
            # fragments from earlier runs sitting in the collection. Clear this
            # file's fragments first, then rebuild them.
            collection.delete(where={"source": file})
            collection.upsert(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            print(f"Memory: Ingested {file} ({len(chunks)} fragments)")
        except Exception as e:
            print(f"Memory Error on {file}: {e}")

    print(f"--- MEMORY: Sync Complete. Total Memories: {collection.count()} ---")

def recall(query, n_results=3):
    """Searches the database for relevant context."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    # Flatten the list of strings
    return results['documents'][0]
