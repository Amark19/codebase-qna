# Codebase Q&A (RAG)

A minimal Retrieval-Augmented Generation tool: index any code repo locally,
then ask natural-language questions about it. Answers are grounded in
retrieved source snippets, with file/chunk references.

## How it works

```
your repo  -->  chunk files  -->  embed chunks (local model)  -->  store in Chroma
                                                                          |
question  -->  embed question  -->  similarity search  --------> top-k chunks
                                                                          |
                                              build prompt with context  |
                                                                          v
                                                                  Claude API
                                                                          |
                                                                          v
                                                                     answer
```

- **Embeddings**: `all-MiniLM-L6-v2` via `sentence-transformers` (runs locally, free, no API key needed for indexing)
- **Vector store**: Chroma (persisted to `./chroma_db`)
- **Generation**: Claude (`claude-sonnet-4-6` via the Anthropic API)
- **Chunking**: simple sliding-window over characters (1500 chars, 200 overlap)

## Setup

```bash
pip install chromadb sentence-transformers anthropic
export ANTHROPIC_API_KEY=your_key_here
```

(The embedding step doesn't need the API key — only `query.py` does, since
that's the part that calls Claude.)

## Usage

### 1. Index a repo

```bash
python index_repo.py /path/to/your/repo
```

This walks the directory, skips junk (`.git`, `node_modules`, `venv`, etc.),
chunks each code/text file, embeds the chunks, and stores everything in a
local Chroma collection called `codebase`.

You can index multiple repos into different collections:

```bash
python index_repo.py /path/to/repo-a repo_a
python index_repo.py /path/to/repo-b repo_b
```

### 2. Ask questions

```bash
python query.py "How does authentication work in this repo?"
python query.py "Where is the database connection configured?" --k 8
python query.py "Explain the chunking logic" --show-chunks
```

`--show-chunks` prints the raw retrieved snippets before the final answer —
useful for understanding *why* Claude answered the way it did, and for
debugging retrieval quality.

## Things to try next (this is where the learning happens)

1. **Chunking strategies** — swap the sliding-window chunker for something
   AST-aware (e.g. split Python files by function/class using the `ast`
   module). Compare answer quality.

2. **Hybrid search** — add a keyword-based retriever (BM25, via `rank_bm25`)
   alongside the embedding search and combine results. Great for queries
   that mention exact variable/function names.

3. **Re-ranking** — retrieve more chunks (k=20) then re-rank with a cross-encoder
   (`sentence-transformers` has these too) before sending the top 5 to Claude.

4. **Bigger/better embeddings** — try `BAAI/bge-base-en-v1.5` or OpenAI's
   `text-embedding-3-small` and compare retrieval quality.

5. **Evaluation** — write a small set of (question, expected_file) pairs and
   check whether the correct file shows up in the top-k results. This is the
   start of a real RAG eval harness.

6. **Streaming + UI** — wrap `query.py` in a Streamlit app so you get a chat
   interface instead of the CLI.

7. **Metadata filtering** — let users restrict search to a subdirectory or
   file type (`collection.query(..., where={"file_path": {"$contains": "src/"}})`).
