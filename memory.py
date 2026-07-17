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
#
# chromadb's default embedder IS all-MiniLM-L6-v2, run through onnxruntime — which
# arrives as one of chromadb's own dependencies. The explicit
# SentenceTransformerEmbeddingFunction that used to be here names the same model
# but reaches it through sentence-transformers, which pulls PyTorch and the whole
# CUDA stack: several GB of dependency to run weights already present.
#
# That extra was never declared in requirements.txt, so this module raised
# ImportError on any clean install — which is why recall() had never once run.
# Same model, same ~256-token ceiling (see core/chunking.py), no CUDA.
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# 3. Get or Create the Collection (The Bookshelf)
collection = client.get_or_create_collection(
    name="enscio_knowledge",
    embedding_function=embedding_function
)





# knowledge/ is shared: the harvester drops source documents here, and
# core/knowledge.py writes the pipeline's own artifacts here too. Indexing our own
# output would feed the master prompt back through retrieval into the local model's
# system prompt — where the master prompt already is. It would appear twice, and
# the store would fill with the pipeline's own claims competing against real
# sources. log.json escapes only because .json is not a readable extension here,
# which is luck rather than design.
PIPELINE_ARTIFACTS = {"master_prompt.txt", "log.json"}


def ingest_knowledge():
    """Reads source documents in 'knowledge/' and memorizes them."""
    print("--- MEMORY: Scanning Knowledge Folder ---")
    folder = "knowledge"
    
    # Clear old memory to avoid duplicates (Simple approach)
    # In production, we would check IDs. For now, we rebuild.
    existing_count = collection.count()
    if existing_count > 0:
        print(f"Memory: Found {existing_count} existing records.")
        # Optional: collection.delete(where={}) # Uncomment to wipe and rebuild
    
    files = [f for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder, f))
             and f not in PIPELINE_ARTIFACTS]

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
