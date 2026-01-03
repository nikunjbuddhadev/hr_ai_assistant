import os
import argparse
from utils import load_model, embed_texts, save_embeddings
from pathlib import Path
from itertools import chain

def read_text_file(path):
    try:
        return Path(path).read_text(encoding='utf-8')
    except Exception as e:
        return ""

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return [c for c in chunks if c]

def main(docs_dir, out_path):
    model = load_model()
    docs = []
    embeddings = []
    metadatas = []
    docs_dir = Path(docs_dir)
    for file in docs_dir.glob('*'):
        if file.is_file():
            text = read_text_file(file)
            chunks = chunk_text(text)
            embs = embed_texts(model, chunks)
            for c, e in zip(chunks, embs):
                docs.append(c)
                embeddings.append(e)
                metadatas.append({"source": str(file.name)})
    save_embeddings(out_path, docs, embeddings, metadatas)
    print(f"Saved {len(docs)} chunks to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs_dir', default='docs', help='Folder with text files')
    parser.add_argument('--out', default='embeddings.json', help='Output embeddings JSON')
    args = parser.parse_args()
    main(args.docs_dir, args.out)
