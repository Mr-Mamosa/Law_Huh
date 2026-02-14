import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import json
from tqdm import tqdm
import torch
import os
import shutil

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_PATH = DATA_DIR / "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "legal_kb"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SAFE_BATCH_SIZE = 64


def load_and_chunk_data(text_splitter):
    documents = []
    ipc_path = DATA_DIR / "ipc_text.txt"
    if ipc_path.exists():
        print(f"Loading and chunking IPC data from {ipc_path}...")
        ipc_content = ipc_path.read_text(encoding="utf-8")
        chunks = text_splitter.split_text(ipc_content)
        for i, chunk in enumerate(chunks):
            documents.append(
                {
                    "id": f"ipc_chunk_{i}",
                    "content": chunk,
                    "metadata": {"source": "Indian Penal Code"},
                }
            )
        print(f"IPC data chunked into {len(chunks)} pieces.")

    sc_jsonl_path = DATA_DIR / "sc_judgments_text.jsonl"
    if sc_jsonl_path.exists():
        print(f"Loading and chunking SC judgments from {sc_jsonl_path}...")
        with open(sc_jsonl_path, "r", encoding="utf-8") as f:
            num_lines = sum(1 for line in f)
            f.seek(0)
            for line in tqdm(
                f, total=num_lines, desc="Reading & Chunking SC Judgments"
            ):
                try:
                    item = json.loads(line)
                    full_text = item.get("text", "")
                    if "Error reading" in full_text or not full_text.strip():
                        continue
                    source_id = item.get("source", "unknown_sc_source")
                    chunks = text_splitter.split_text(full_text)
                    for i, chunk in enumerate(chunks):
                        documents.append(
                            {
                                "id": f"sc_chunk_{source_id}_{i}",
                                "content": chunk,
                                "metadata": {
                                    "source": f"Supreme Court Judgment: {source_id}"
                                },
                            }
                        )
                except Exception as e:
                    print(f"
Skipping a line due to an error: {e}")

    return documents


def add_documents_to_chroma(client, documents, embedding_model):
    print(f"
Checking for existing collection '{COLLECTION_NAME}'...")
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(
            f"Existing collection '{COLLECTION_NAME}' deleted to ensure a clean start."
        )
    except ValueError:
        print(f"Collection '{COLLECTION_NAME}' did not exist. Creating new one.")
    except Exception as e:
        print(f"Note: Cleanup check encountered: {e}")

    print(f"Creating fresh ChromaDB collection: {COLLECTION_NAME}")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:batch_size": 10000
        },
    )

    print(
        f"Starting ingestion of {len(documents)} documents in batches of {SAFE_BATCH_SIZE}..."
    )

    for i in tqdm(range(0, len(documents), SAFE_BATCH_SIZE), desc="Ingesting Batches"):
        batch = documents[i : i + SAFE_BATCH_SIZE]
        ids = [doc["id"] for doc in batch]
        contents = [doc["content"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]

        try:
            embeddings = embedding_model.encode(
                contents,
                show_progress_bar=False,
                batch_size=SAFE_BATCH_SIZE,
                convert_to_tensor=False,
            ).tolist()

            collection.add(
                embeddings=embeddings, documents=contents, metadatas=metadatas, ids=ids
            )
        except Exception as e:
            print(
                f"
[ERROR] Failed to process batch starting at index {i}. Skipping batch. Error: {e}"
            )

    print("
Ingestion complete.")
    count = collection.count()
    print(f"The collection '{COLLECTION_NAME}' now contains {count} documents.")


def setup_db():
    print('Starting ChromaDB setup...')
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA detected. Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(f"CUDA not available. Using device: {device}")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    print("Embedding model loaded.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    all_documents = load_and_chunk_data(text_splitter)
    if all_documents:
        add_documents_to_chroma(client, all_documents, embedding_model)
    else:
        print("No documents were found to process.")
    print('ChromaDB setup complete.')


if __name__ == "__main__":
    setup_db()
