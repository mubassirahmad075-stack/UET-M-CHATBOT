# UETM Assistant — UET Mardan

A RAG (retrieval-augmented generation) chatbot that answers questions about the
University of Engineering & Technology (UET) Mardan using the official
prospectus. It retrieves the most relevant passages from a local vector index
and answers **strictly from that context** — no invented facts. If the answer is
not in the dataset, it tells the user to contact the Directorate of Admissions.

## Features

- Answers in English, Urdu, or Roman Urdu (matches the student's language)
- Knows about admissions, fees, departments, grading rules, hostels, and exams
- **Forms guide**: a built-in catalog of official application forms (admission,
  correction, migration, hostel, semester registration, freeze, rechecking,
  transcripts, grievances) — the assistant tells students exactly which form
  applies to what, where to submit it, and the fee
- **Downloads catalog**: every official file from the UET Mardan website
  downloads section (76 files: forms, acts, statutes, budgets, rules, policies,
  calendars, newsletters, proformas) with direct links — ask "where can I
  download X?" or browse the ⬇️ Downloads panel
- No raw dataset pages shown to users
- Polished UI with the UET green palette and subtle 3D animations

## Local setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

# 1. Add your OpenAI key
copy .env.example .env         # Windows
# cp .env.example .env         # macOS / Linux
# then edit .env and paste your OPENAI_API_KEY

# 2. Build the vector index from the PDFs (one-time, needs the key)
python ingest.py

# 3. Run the app
python app.py
```

Open http://127.0.0.1:5000

## Deployment (production)

The app is a standard Flask WSGI application.

### Option A — Docker

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# index/ and .env are mounted at runtime (see below)
EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]
```

```bash
docker build -t uetm-assistant .
docker run -d -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -v /path/to/index:/app/index \
  uetm-assistant
```

### Option B — VM / VPS (gunicorn)

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

Put a reverse proxy (nginx/caddy) in front with HTTPS.

### Option C — PaaS (Render / Railway / Fly.io)

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn -w 2 -b 0.0.0.0:$PORT app:app`
- Set env var `OPENAI_API_KEY` (and optionally `CHAT_MODEL`)
- **Important**: the `index/` folder must exist at runtime — either commit it
  (it is gitignored by default) or run `python ingest.py` in a pre-deploy job.

## Environment variables

| Variable      | Required | Default       | Description            |
|---------------|----------|---------------|------------------------|
| `OPENAI_API_KEY` | Yes   | —             | OpenAI API key         |
| `CHAT_MODEL`     | No    | `gpt-4o-mini` | Chat model for answers |

## Project layout

```
app.py               Flask app: retrieval + chat + forms/downloads catalogs
ingest.py            Builds index/ from the PDFs (embeddings)
index/               Vector index (chunks.json + vectors.npy) — build with ingest.py
static/index.html    Chat UI (green theme, 3D animation, forms + downloads panels)
downloads_catalog.md Reference catalog of all official UET Mardan website downloads
requirements.txt     Python dependencies
.env.example         Template for your API key (never commit .env)
```

## Notes

- The index is built with `text-embedding-3-small` and stores L2-normalized
  vectors, so retrieval is cosine similarity.
- Answers are generated at low temperature (0.2) to stay faithful to the
  prospectus.
