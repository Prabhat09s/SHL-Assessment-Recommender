# SHL Assessment Recommender

A conversational agent that recommends SHL Individual Test Solutions
through dialogue, per the take-home assignment spec. FastAPI service
exposing `GET /health` and `POST /chat`.

## API key required

**Groq** — free tier, no credit card. Get one at
https://console.groq.com -> API Keys, then set it as `GROQ_API_KEY`
(see `.env.example`). No other API key is needed; this project does not
call OpenAI, Anthropic, or any paid service.

If `GROQ_API_KEY` is not set, the server still starts and `/health` /
`/chat` still respond (schema-correct), but `/chat` will just tell you
it isn't connected to a model instead of holding a real conversation.

## How it works

1. **Catalog (`data/catalog.json`)** — scraped SHL Individual Test
   Solutions. Each item has name, url, test_type (`keys`), duration,
   languages, job_levels, description.
   **The included file is a 44-item sample** covering every assessment
   referenced in the 10 provided conversation traces, plus some extras,
   so you can run and test immediately. **Replace it with your full
   scraped catalog** (same JSON schema) before submitting — the
   assignment requires "the entire SHL catalogue."

2. **Retrieval (`app/catalog.py`, `app/retrieval.py`)** — TF-IDF +
   cosine similarity over name/description/keys/job_levels. No vector
   DB needed at this catalog size; fast enough to stay well inside the
   30-second per-call budget.

3. **Agent (`app/agent.py`)** — sends the full conversation plus the
   top-K retrieved catalog candidates to a Groq LLM, with a system
   prompt that encodes the required behaviors (clarify / recommend /
   refine / compare / refuse) and anti-hallucination rules. The model
   is asked to reply with a single JSON object in the exact response
   schema.

4. **Validation (`app/agent.py: _parse_and_validate`)** — a
   post-processing guardrail: any recommended item whose `url` doesn't
   match something actually in the catalog is silently dropped before
   the response goes out. This is the hard backstop against
   hallucinated assessments, on top of prompt instructions.

5. **API (`app/main.py`)** — thin FastAPI layer; stateless by design
   (no session storage — every `/chat` call gets the full history and
   is processed independently, per the assignment spec).

## Local setup

```bash
cd shl-recommender
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your real GROQ_API_KEY
uvicorn app.main:app --reload --port 8000
```

Check it's alive:

```bash
curl http://localhost:8000/health
```

Try a conversation:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"We are hiring a senior Java backend engineer, 5+ years, works with Spring and SQL."}]}'
```

Run the smoke test (no API key needed — exercises the FastAPI app,
catalog loading, and retrieval directly):

```bash
python test_smoke.py
```

## Deployment

Any platform that runs a Dockerfile or a plain Python web service works
(Render, Fly.io, Railway, Modal, Hugging Face Spaces, etc.). Using the
included Dockerfile:

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key_here shl-recommender
```

On your hosting platform: set `GROQ_API_KEY` as an environment
variable/secret (never commit it), and make sure both `/health` and
`/chat` are reachable at your public URL before submitting.

## Known limitations / things to state in your approach doc

- **Sample catalog only.** Swap in your full scrape before submission
  and re-run `test_smoke.py` to confirm it still loads and indexes.
- **Retrieval is TF-IDF, not embeddings.** This is a deliberate
  simplicity/latency trade-off for a catalog of a few thousand items —
  worth justifying (or upgrading to embeddings) in your write-up.
- **Recommendations always return as `[]`, never `null`**, matching the
  literal JSON example in the assignment PDF (the "null" language in
  the trace commentary is descriptive, not literal schema).
- **Report bundling and Job Solutions scoping** (e.g. whether items
  like "Entry Level Customer Serv - Retail & Contact Center" count as
  in-scope "Individual Test Solutions") is genuinely ambiguous in the
  source traces vs. the PDF's stated scope — the system prompt allows
  bundling a report with its base test, but you should explicitly note
  your scoping decision in the approach doc since it affects Recall@10.
- **No conversation-turn state is stored server-side.** The agent
  re-derives the "current shortlist" purely by reading the conversation
  history each call, per the stateless API requirement.
