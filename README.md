# AI Pulse — AI news, summarized

A static AI news website. Each hour it collects headlines from about 14 AI news feeds. Then an LLM groups duplicate coverage into one story, writes a one-line summary, a "why it matters" line and topic tags for each story, and picks the Top 5. Every headline links to the original publisher.

## How it works

```
RSS/Atom feeds ──► fetch_news.py ──► raw.json ──► enrich.py (Claude) ──► news.json ──► index.html
                                                     │
                                                state.json (cache: only NEW articles go to the LLM)
```

| File                                | Purpose                                                                                                                              |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `fetch_news.py`                     | Reads the feeds in `FEEDS`, cleans the text, removes exact duplicates, writes `raw.json`                                             |
| `enrich.py`                         | AI step. Runs (1) grouping, (2) summaries, why-it-matters lines and topics, (3) Top 5. Writes `news.json` and the `state.json` cache |
| `index.html`                        | The website: Top 5, latest feed, topic chips, search, grouped sources, dark mode                                                     |
| `.github/workflows/update-news.yml` | Runs both scripts every hour and commits the results                                                                                 |

### The AI step in a bit more detail

- **Structured output:** every LLM call uses a forced tool call with a JSON schema, so the replies always have the right shape. Topics come from a fixed list (an enum).
- **Caching:** each article is processed once. A typical hourly run sends about 10–30 new articles, which takes 2–3 calls to a small model.
- **Safe fallback:** if an API call fails or `ANTHROPIC_API_KEY` is missing, that step falls back to heuristics (word-overlap grouping, the feed's first sentence as the summary, keyword topics, "most outlets" for the Top 5). The site keeps updating either way.
- **Prompt-injection guard:** feed text is treated as untrusted data in the system prompt. Output is validated (story IDs must exist, topics must be in the list), and the page HTML-escapes everything it shows.
- **Model:** `claude-haiku-5-5` by default. To change it, set the `NEWS_MODEL` environment variable.

## Run locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # optional; without it, heuristic mode is used
python fetch_news.py && python enrich.py
python -m http.server 8000               # open http://localhost:8000
```

## Deploy free on GitHub Pages

1. Create a repo and push these files.
2. Settings → Secrets and variables → Actions → New secret: `ANTHROPIC_API_KEY`.
3. Settings → Pages → Deploy from branch → `main` / root.
4. Actions → "Update AI news" → Run workflow. After that it runs every hour.

The `news.json` included here is sample data, so the page renders before the first run. The first workflow run replaces it.

## Customize

- **Sources:** edit `FEEDS` in `fetch_news.py` as `(name, category, url)` entries.
- **Topics:** edit `TOPICS` in `enrich.py`. The chips on the site update automatically.
- **Time windows:** `WINDOW_HOURS` (how long stories stay on the site, default 72) and `TOP_WINDOW_HOURS` (default 24).

## Ideas for later

- A daily email or Slack digest of the Top 5
- Embedding-based search ("show me everything about agent evals")
- Moving to FastAPI + Postgres on AWS for user accounts, saved stories and personalized feeds
