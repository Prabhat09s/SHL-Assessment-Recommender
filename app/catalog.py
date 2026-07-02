"""
Loads the scraped SHL catalog and provides keyword-grounded retrieval
over it. Retrieval uses TF-IDF + cosine similarity, which is fast,
dependency-light, and fully explainable (no vector DB needed for a
catalog of this size).
"""
import json
import re
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    # collapse stray newlines/whitespace (some scraped fields contain
    # literal line breaks inside a single field - see entity 4207)
    return " ".join(str(value).split())


class Catalog:
    def __init__(self, path: str):
        self.items: List[Dict] = self._load(path)
        self._build_index()

    # ---------- loading ----------

    def _load(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        cleaned = []
        seen_urls = set()
        for item in raw:
            name = _clean_text(item.get("name"))
            link = _clean_text(item.get("link"))
            if not name or not link:
                continue
            if link in seen_urls:
                continue  # de-dupe
            seen_urls.add(link)

            item["name"] = name
            item["link"] = link
            item["description"] = _clean_text(item.get("description"))
            item["keys"] = item.get("keys") or []
            item["job_levels"] = item.get("job_levels") or []
            item["languages"] = item.get("languages") or []
            item["duration"] = _clean_text(item.get("duration"))
            cleaned.append(item)
        return cleaned

    # ---------- indexing ----------

    def _doc_text(self, item: Dict) -> str:
        parts = [
            item.get("name", ""),
            item.get("name", ""),  # weight name x2
            item.get("description", ""),
            " ".join(item.get("keys", [])),
            " ".join(item.get("job_levels", [])),
        ]
        return " ".join(p for p in parts if p)

    def _build_index(self):
        self.corpus = [self._doc_text(i) for i in self.items]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=30000,
        )
        if self.corpus:
            self.matrix = self.vectorizer.fit_transform(self.corpus)
        else:
            self.matrix = None

    # ---------- lookup ----------

    def by_url(self, url: str) -> Optional[Dict]:
        url = _clean_text(url)
        for item in self.items:
            if item.get("link") == url:
                return item
        return None

    def by_name_fuzzy(self, name: str) -> Optional[Dict]:
        norm = re.sub(r"[^a-z0-9]", "", name.lower())
        for item in self.items:
            if re.sub(r"[^a-z0-9]", "", item["name"].lower()) == norm:
                return item
        return None

    def search(self, query: str, top_k: int = 18) -> List[Tuple[Dict, float]]:
        if not query.strip() or self.matrix is None:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix).flatten()
        ranked_idx = sims.argsort()[::-1][:top_k]
        results = []
        for idx in ranked_idx:
            score = float(sims[idx])
            if score <= 0:
                continue
            results.append((self.items[idx], score))
        return results

    def __len__(self):
        return len(self.items)
