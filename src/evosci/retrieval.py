from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .config import RetrievalConfig


@dataclass(slots=True)
class Paper:
    title: str
    abstract: str
    url: str
    published: str

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "url": self.url,
            "published": self.published,
        }


class LiteratureRetriever:
    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    def search(self, query: str) -> list[Paper]:
        if not self.config.enabled or self.config.max_results == 0:
            return []
        if self.config.provider != "arxiv":
            raise ValueError(f"Unsupported retrieval provider: {self.config.provider}")
        return self._search_arxiv(query)

    def _search_arxiv(self, query: str) -> list[Paper]:
        params = urllib.parse.urlencode({
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": self.config.max_results,
            "sortBy": "relevance",
        })
        request = urllib.request.Request(
            "https://export.arxiv.org/api/query?" + params,
            headers={"User-Agent": "EvoSciReproduction/0.1"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                document = ET.fromstring(response.read())
        except (OSError, ET.ParseError):
            return []
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        results = []
        for entry in document.findall("atom:entry", namespace):
            title = " ".join((entry.findtext("atom:title", "", namespace)).split())
            abstract = " ".join((entry.findtext("atom:summary", "", namespace)).split())
            url = entry.findtext("atom:id", "", namespace)
            published = entry.findtext("atom:published", "", namespace)
            results.append(Paper(title, abstract, url, published))
        return results
