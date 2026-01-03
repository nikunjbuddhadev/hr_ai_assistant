import streamlit as st
import os
from utils import load_model, embed_texts, cosine_search, load_embeddings
import numpy as np
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

EMB_PATH = 'embeddings.json'

st.set_page_config(page_title='HR AI Assistant', layout='wide')
st.title('HR AI Assistant (Tech-AI-Thon Demo)')

col1, col2 = st.columns([3,1])

with col2:
    st.header('Admin')
    uploaded = st.file_uploader('Upload .txt HR doc', type=['txt'])
    if uploaded is not None:
        save_path = Path('docs') / uploaded.name
        save_path.parent.mkdir(exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(uploaded.getvalue())
        st.success(f'Saved {uploaded.name}. Run ingest.py to update embeddings or restart app.')
    if Path(EMB_PATH).exists():
        st.write('Embeddings loaded.')
    else:
        st.warning('No embeddings found. Run `python ingest.py --docs_dir docs` to create embeddings.')

with col1:
    st.header('Ask HR')
    query = st.text_input('Ask a question about HR policies:')
    top_k = st.slider('Top relevant snippets', 1, 5, 3)
    if st.button('Get Answer') and query.strip():
        if not Path(EMB_PATH).exists():
            st.error('Embeddings not found. Please run ingest.py first.')
        else:
            docs, embs, metas = load_embeddings(EMB_PATH)
            model = load_model()
            q_emb = model.encode([query])[0].tolist()
            idxs, scores = cosine_search(q_emb, embs, top_k)
            snippets = []
            for i,s in zip(idxs,scores):
                snippets.append(f"Source: {metas[i]['source']}\nScore: {s:.3f}\n{docs[i]}\n---\n")
            context = "\n".join(snippets)
            st.subheader('Retrieved Snippets')
            for sn in snippets:
                st.write(sn)
            # If OPENAI_API_KEY is set, call OpenAI to generate friendly answer
            OPENAI_KEY = os.environ.get('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
            if OPENAI_KEY:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_KEY)
                prompt = f"You are an HR assistant. Answer the question using ONLY the context below. If the answer cannot be found, say 'This information is not available in the documents.'\n\nContext:\n{context}\nQuestion: {query}\nAnswer:"
                try:
                    resp = client.chat.completions.create(
                        model=os.environ.get('OPENAI_MODEL', 'gpt-4o'),
                        messages=[{'role':'user','content':prompt}],
                        max_tokens=400,
                        temperature=0.0
                    )
                    answer = resp.choices[0].message.content.strip()
                except Exception as e:
                    answer = f"Error calling OpenAI: {e}\n\nFalling back to showing context only.\n\n{context}"
            else:
                answer = "\n".join([s for s in snippets])
                answer = "OpenAI API key not set. Showing retrieved context instead:\n\n" + answer
            st.subheader('Answer')
            st.write(answer)
