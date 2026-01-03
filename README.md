# HR AI Assistant (Hackathon-ready)

**What this is**
A lightweight HR AI Assistant that answers employee queries using your internal HR documents.
- Retrieval-based: finds relevant document snippets using embeddings.
- Optional LLM answer: if you provide an `OPENAI_API_KEY`, the app will use OpenAI to generate a natural answer based on retrieved context.
- Built with Streamlit for quick demo.

**How to run (locally)**
1. Create a Python environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. Ingest sample documents (already present) or add your own:
   ```bash
   python ingest.py --docs_dir docs
   ```
   This will create an embeddings file `embeddings.json` in the project folder.

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. (Optional) To enable AI-generated answers, set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   # or create a .env file with OPENAI_API_KEY=...
   ```

**Files**
- `app.py` - Streamlit app for chat and admin upload
- `ingest.py` - Script to create embeddings from documents
- `utils.py` - Embedding and retrieval helpers
- `docs/` - sample HR documents (text files)
- `embeddings.json` - generated after running ingest.py
- `requirements.txt` - Python deps

**Notes / Limitations**
- This project uses `sentence-transformers` for embeddings.
- OpenAI integration is optional. Without an API key, the app returns the most relevant snippets and indicates whether the exact answer exists in documents.
- For production, consider using a managed vector DB (Pinecone/Chroma), better chunking, authentication, and document access controls.

Enjoy the hackathon! 🎉
