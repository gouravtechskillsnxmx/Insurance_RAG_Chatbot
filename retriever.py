import os, json, numpy as np
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from models import Document

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = None
try:
    client = OpenAIClient(api_key=OPENAI_API_KEY)
except Exception:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        client = openai
    except Exception:
        client = None

INDEX_DIR = './vector_index'
os.makedirs(INDEX_DIR, exist_ok=True)
DOCS_PATH = os.path.join(INDEX_DIR, 'docs.json')
EMB_PATH = os.path.join(INDEX_DIR, 'embeddings.npy')

def embed_texts(texts):
    if client is None:
        raise RuntimeError('OpenAI client not initialized for embeddings.')
    try:
        res = client.embeddings.create(model=os.getenv('OPENAI_EMBED_MODEL','text-embedding-3-small'), input=texts)
        return [d.embedding for d in res.data]
    except Exception:
        res = client.Embedding.create(model='text-embedding-3-small', input=texts)
        return [r['embedding'] for r in res['data']]

def ingest_documents(engine=None, data_dir='./rag/data'):
    if engine is None:
        engine = create_engine(os.getenv('DATABASE_URL','sqlite:///./insurance_dashboard.db'), future=True)
    SessionLocal = sessionmaker(bind=engine, future=True)
    all_rows = []
    for fn in os.listdir(data_dir):
        if fn.endswith('.md') or fn.endswith('.txt'):
            with open(os.path.join(data_dir, fn), 'r', encoding='utf-8') as f:
                content = f.read()
            # simple chunking
            size=1200; overlap=200; start=0; idx=0
            while start < len(content):
                end = min(len(content), start+size)
                chunk = content[start:end]
                all_rows.append({'title': f"{fn}#chunk-{idx}",'content': chunk})
                idx += 1; start = end - overlap
    # embed
    texts = [r['content'] for r in all_rows]
    batch = 64
    vecs = []
    for i in range(0, len(texts), batch):
        vecs.extend(embed_texts(texts[i:i+batch]))
    SessionLocal = sessionmaker(bind=engine, future=True)
    with SessionLocal() as s:
        for r,v in zip(all_rows, vecs):
            doc = Document(title=r['title'], content=r['content'], meta_data={}, embedding=v)
            s.add(doc)
        s.commit()
    # save index
    np.save(EMB_PATH, np.array(vecs, dtype='float32'))
    with open(DOCS_PATH, 'w', encoding='utf-8') as f:
        json.dump([{'title': r['title'], 'content': r['content']} for r in all_rows], f, ensure_ascii=False)
    return len(all_rows)

def _load_index():
    if not os.path.exists(DOCS_PATH) or not os.path.exists(EMB_PATH):
        return [], None
    with open(DOCS_PATH, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    emb = np.load(EMB_PATH)
    norms = np.linalg.norm(emb, axis=1)
    return docs, (emb, norms)

def search_similar(query, k=5):
    docs, data = _load_index()
    if not docs: return []
    emb, norms = data
    vec = np.array(embed_texts([query])[0], dtype='float32')
    qn = np.linalg.norm(vec) + 1e-10
    sims = (emb @ vec) / (norms * qn + 1e-10)
    idx = np.argsort(-sims)[:k]
    return [docs[i] for i in idx if 0<=i<len(docs)]
