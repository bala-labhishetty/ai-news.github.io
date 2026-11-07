"""Fetch AI news from RSS/Atom feeds and write raw.json (input for enrich.py).

Usage:  python fetch_news.py            # writes raw.json next to this file
        python fetch_news.py --max 400  # keep up to 400 items
"""
import argparse
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import feedparser

# (source name, category, feed URL) — edit freely.
FEEDS = [
    # AI labs
    ("OpenAI", "AI Labs", "https://openai.com/news/rss.xml"),
    ("Google DeepMind", "AI Labs", "https://deepmind.google/blog/rss.xml"),
    ("Google AI", "AI Labs", "https://blog.google/technology/ai/rss/"),
    ("Hugging Face", "Open Source", "https://huggingface.co/blog/feed.xml"),
    ("NVIDIA", "AI Labs", "https://blogs.nvidia.com/feed/"),
    ("AWS ML Blog", "Cloud & Enterprise", "https://aws.amazon.com/blogs/machine-learning/feed/"),
    # News
    ("TechCrunch", "Industry", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("The Verge", "Industry", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
    ("VentureBeat", "Industry", "https://venturebeat.com/category/ai/feed/"),
    ("MIT Tech Review", "Industry", "https://www.technologyreview.com/topic/artificial-intelligence/feed"),
    ("Ars Technica", "Industry", "https://feeds.arstechnica.com/arstechnica/technology-lab"),
    # Community & research
    ("Hacker News", "Community", "https://hnrss.org/newest?q=AI+OR+LLM&points=50"),
    ("arXiv cs.AI", "Research", "https://rss.arxiv.org/rss/cs.AI"),
    ("arXiv cs.CL", "Research", "https://rss.arxiv.org/rss/cs.CL"),
]

UA = "ai-news-aggregator/1.0 (+https://github.com/)"


def clean_text(s: str, limit: int = 800) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = re.sub(r"\s+", " ", html.unescape(s)).strip()
    return s[:limit]


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def entry_time(e) -> datetime:
    for key in ("published_parsed", "updated_parsed"):
        tm = e.get(key)
        if tm:
            return datetime(*tm[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def parse_feed(source: str, category: str, url_or_path: str) -> list[dict]:
    feed = feedparser.parse(url_or_path, agent=UA)
    if feed.bozo and not feed.entries:
        print(f"  ! {source}: {feed.get('bozo_exception')}", file=sys.stderr)
        return []
    items = []
    for e in feed.entries:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        if not title or not link:
            continue
        items.append({
            "title": re.sub(r"\s+", " ", title),
            "link": link,
            "source": source,
            "category": category,
            "published": entry_time(e).isoformat(),
            "description": clean_text(e.get("summary") or e.get("description") or ""),
        })
    print(f"  {source}: {len(items)} items")
    return items


def dedupe(items: list[dict]) -> list[dict]:
    seen_links, seen_titles, out = set(), set(), []
    for it in sorted(items, key=lambda x: x["published"], reverse=True):
        link_key = it["link"].split("?")[0].rstrip("/")
        title_key = norm_title(it["title"])
        if link_key in seen_links or title_key in seen_titles:
            continue
        seen_links.add(link_key)
        seen_titles.add(title_key)
        out.append(it)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--out", default=str(Path(__file__).with_name("raw.json")))
    args = ap.parse_args(argv)

    all_items = []
    for source, category, url in FEEDS:
        try:
            all_items += parse_feed(source, category, url)
        except Exception as exc:  # one bad feed must not break the run
            print(f"  ! {source}: {exc}", file=sys.stderr)

    items = dedupe(all_items)[: args.max]
    if not items:
        print("No items fetched; keeping existing raw.json", file=sys.stderr)
        return 1

    payload = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "count": len(items),
        "items": items,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"Wrote {len(items)} items to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
