import os, io, json
from datetime import datetime
from functools import wraps
from flask import Flask, request, render_template, redirect, url_for, session, send_file, flash, jsonify
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

load_dotenv()

from models import init_db, Lead, Chat, Document
from retriever import search_similar, ingest_documents, embed_texts

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise RuntimeError('OPENAI_API_KEY not set. Put it in .env or environment.')

# Initialize OpenAI client (new SDK preferred, fallback to old)
try:
    from openai import OpenAI as OpenAIClient
    client = OpenAIClient(api_key=OPENAI_API_KEY)
    SDK_MODE = 'new'
except Exception:
    import openai
    openai.api_key = OPENAI_API_KEY
    client = openai
    SDK_MODE = 'old'

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret')
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./insurance_dashboard.db')
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, future=True)
init_db(engine)

def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if session.get('logged_in'): return f(*args, **kwargs)
        return redirect(url_for('login'))
    return wrapped

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        if request.form.get('username') == os.getenv('ADMIN_USER') and request.form.get('password') == os.getenv('ADMIN_PASS'):
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        flash('Invalid credentials','danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear(); return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    q = request.args.get('q','').strip()
    with SessionLocal() as s:
        leads = s.execute(select(Lead).order_by(Lead.created_at.desc())).scalars().all()
    if q:
        leads = [l for l in leads if q in (l.phone or '') or q.lower() in ((l.name or '') + (l.email or '') + (l.interest or '')).lower()]
    return render_template('dashboard.html', leads=leads, q=q)

@app.route('/export')
@login_required
def export_csv():
    import csv
    with SessionLocal() as s:
        leads = s.execute(select(Lead)).scalars().all()
    mem = io.StringIO()
    w = csv.writer(mem)
    w.writerow(['id','name','email','phone','interest','created_at'])
    for l in leads:
        w.writerow([l.id, l.name, l.email, l.phone, l.interest, l.created_at.isoformat()])
    mem.seek(0)
    return send_file(io.BytesIO(mem.read().encode('utf-8')), mimetype='text/csv', download_name='leads.csv')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json or {}
    message = data.get('message','').strip()
    lead = data.get('lead', {})
    phone = (lead.get('phone') or '').strip()
    if not phone:
        return jsonify({'error':'Phone number (mobile) is mandatory'}), 400
    # store lead
    with SessionLocal() as s:
        l = Lead(name=lead.get('name'), email=lead.get('email'), phone=phone, interest=lead.get('interest'))
        s.add(l); s.commit(); s.refresh(l)
        lead_id = l.id
    # retrieve context
    contexts = search_similar(message, k=5)
    context_blob = "\n\n---\n".join([f"Title: {c.get('title')}\n{c.get('content')[:1000]}" for c in contexts])
    system = "You are an expert Indian insurance sales assistant for licensed agents. Use the context to answer precisely."
    messages = [{"role":"system","content": system + ("\nContext:\n"+context_blob if context_blob else "")},{"role":"user","content": message}]
    # call OpenAI (new vs old)
    if SDK_MODE == 'new':
        resp = client.chat.completions.create(model=os.getenv('OPENAI_CHAT_MODEL','gpt-4o-mini'), messages=messages, temperature=0.2)
        answer = resp.choices[0].message.content
    else:
        resp = client.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages)
        answer = resp.choices[0]['message']['content']
    # save chat
    with SessionLocal() as s:
        s.add(Chat(lead_id=lead_id, role='user', content=message))
        s.add(Chat(lead_id=lead_id, role='assistant', content=answer))
        s.commit()
    return jsonify({'answer': answer, 'context_used':[c.get('title') for c in contexts], 'lead_id': lead_id})

@app.route('/admin/ingest', methods=['POST'])
@login_required
def admin_ingest():
    count = ingest_documents(engine=engine)
    return jsonify({'ingested': count})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
