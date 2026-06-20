"""
Index a code repository into a Chroma vector store for RAG-based Q&A.

Usage:
    python index_repo.py /path/to/repo
"""

import os
import sys
import hashlib
import chromadb
from tfidf_embeddings import TfidfEmbeddingFunction

# File extensions worth indexing
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".md", ".txt",
    ".json", ".yaml", ".yml", ".toml",
}

# Directories to skip entirely
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", "target", "vendor",
}

CHUNK_SIZE = 1500      # characters per chunk
CHUNK_OVERLAP = 200    # overlap between chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window character-based chunking."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def walk_repo(repo_path):
    """Yield (filepath, content) for all relevant code files."""
    for root, dirs, files in os.walk(repo_path):
        # prune skip dirs in-place so os.walk doesn't descend into them
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in CODE_EXTENSIONS:
                continue

            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                print(f"  ! skipping {fpath}: {e}")
                continue

            if content.strip():
                yield fpath, content


def main():
    if len(sys.argv) < 2:
        print("Usage: python index_repo.py /path/to/repo [collection_name]")
        sys.exit(1)

    repo_path = os.path.abspath(sys.argv[1])
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "codebase"

    if not os.path.isdir(repo_path):
        print(f"Error: {repo_path} is not a directory")
        sys.exit(1)

    print(f"Indexing repo: {repo_path}")
    print(f"Collection: {collection_name}")

    # Set up Chroma with a local TF-IDF embedder (fully offline, no downloads)
    client = chromadb.PersistentClient(path="./chroma_db")

    ids, documents, metadatas = [], [], []
    file_count = 0
    chunk_count = 0

    for fpath, content in walk_repo(repo_path):
        rel_path = os.path.relpath(fpath, repo_path)
        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{rel_path}:{i}".encode()).hexdigest()
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "file_path": rel_path,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
            chunk_count += 1

        file_count += 1
        print(f"  + {rel_path} ({len(chunks)} chunk{'s' if len(chunks) != 1 else ''})")

    if not documents:
        print("No files found to index. Check your path and extensions.")
        sys.exit(1)

    # Fit TF-IDF vectorizer on the full corpus, then save it so query.py
    # can use the exact same vocabulary/vectors at query time.
    print("\nFitting TF-IDF vectorizer on corpus...")
    embedding_fn = TfidfEmbeddingFunction()
    embedding_fn.fit(documents)
    embedding_fn.save(os.path.join("chroma_db", f"{collection_name}_tfidf.pkl"))

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )

    # Add in batches (Chroma has batch size limits for some backends)
    BATCH = 100
    print(f"\nEmbedding and storing {chunk_count} chunks from {file_count} files...")
    for i in range(0, len(documents), BATCH):
        collection.add(
            ids=ids[i:i + BATCH],
            documents=documents[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )

    print(f"\nDone! Indexed {file_count} files / {chunk_count} chunks into '{collection_name}'.")
    print("Now run: python query.py \"your question here\"")


if __name__ == "__main__":
    main()
