import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# Chunking is pure logic and lives in core/chunking.py so it can be tested
# without loading a vector database. CHUNK_SIZE carries the embedder's
# ~256-token ceiling — see that module before changing it.
from core.chunking import CHUNK_OVERLAP, CHUNK_SIZE, split_text

# ── Embedding / chunking configuration ────────────────────────────────────────
# memory.py is a standalone RAG tool, not part of the core/ request path, so it
# owns its own model config rather than importing core.clients (which would drag
# in the paid API clients and require their keys just to index a folder).
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


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
