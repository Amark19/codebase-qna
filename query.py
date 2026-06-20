"""
Ask questions about an indexed codebase using RAG + Claude.

Usage:
    python query.py "How does authentication work in this repo?"
    python query.py "Explain the chunking logic" --collection codebase --k 5
"""

import sys
import argparse
import os
import chromadb
from tfidf_embeddings import TfidfEmbeddingFunction
import anthropic


def retrieve(question, collection_name="codebase", k=5):
    client = chromadb.PersistentClient(path="./chroma_db")

    tfidf_path = os.path.join("chroma_db", f"{collection_name}_tfidf.pkl")
    if not os.path.exists(tfidf_path):
        print(f"No TF-IDF model found for collection '{collection_name}'. Run index_repo.py first.")
        sys.exit(1)
    embedding_fn = TfidfEmbeddingFunction.load(tfidf_path)

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )
    except Exception:
        print(f"Collection '{collection_name}' not found. Run index_repo.py first.")
        sys.exit(1)

    results = collection.query(
        query_texts=[question],
        n_results=k,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    return list(zip(docs, metas, distances))


def build_prompt(question, retrieved_chunks):
    context_blocks = []
    for doc, meta, dist in retrieved_chunks:
        header = f"--- {meta['file_path']} (chunk {meta['chunk_index']+1}/{meta['total_chunks']}) ---"
        context_blocks.append(f"{header}\n{doc}")

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant answering questions about a codebase.
Use ONLY the following code context to answer the question. If the context
doesn't contain enough information, say so explicitly rather than guessing.
When relevant, mention which file(s) your answer is based on.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    return prompt


def ask_claude(prompt):
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question to ask about the codebase")
    parser.add_argument("--collection", default="codebase", help="Chroma collection name")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--show-chunks", action="store_true", help="Print retrieved chunks before the answer")
    args = parser.parse_args()

    print(f"Searching for relevant code...\n")
    retrieved = retrieve(args.question, args.collection, args.k)

    if args.show_chunks:
        print("=== Retrieved chunks ===")
        for doc, meta, dist in retrieved:
            print(f"\n[{meta['file_path']}] (distance: {dist:.4f})")
            print(doc[:300] + ("..." if len(doc) > 300 else ""))
        print("\n========================\n")

    print("Top matches:")
    for doc, meta, dist in retrieved:
        print(f"  - {meta['file_path']} (chunk {meta['chunk_index']+1}/{meta['total_chunks']}, distance {dist:.4f})")

    prompt = build_prompt(args.question, retrieved)

    print("\nAsking Claude...\n")
    answer = ask_claude(prompt)

    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
