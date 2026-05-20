from __future__ import annotations

import re
from typing import Iterator
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from config import REQUEST_HEADERS, TUTORTOP_BASE, TUTORTOP_CATEGORY
from .base import BaseCollector


PRICE_RE = re.compile(r"(\d[\d\s]{1,8})\s*(?:₽|руб|р\.)", re.IGNORECASE)
MONTHS_RE = re.compile(r"(\d+)\s*мес", re.IGNORECASE)
HOURS_RE = re.compile(r"(\d+)\s*час", re.IGNORECASE)


class TutortopCollector(BaseCollector):
    source_name = "tutortop"

    def __init__(self, cache_path, refresh: bool = False, max_pages: int = 3):
        super().__init__(cache_path, refresh=refresh, throttle=1.2)
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _get(self, url: str) -> str:
        r = self.session.get(url, timeout=20)
        r.raise_for_status()
        return r.text

    def _fetch(self) -> Iterator[dict]:
        for page in range(1, self.max_pages + 1):
            url = urljoin(TUTORTOP_BASE, TUTORTOP_CATEGORY)
            if page > 1:
                url = f"{url}?page={page}"
            html = self._get(url)
            soup = BeautifulSoup(html, "lxml")

            cards = soup.select("div.course-card, article.course-item, li.catalog-item")
            if not cards:
                # запасной селектор на случай редизайна
                cards = soup.find_all(attrs={"data-course": True})

            for card in cards:
                row = self._parse_card(card)
                if row:
                    yield row

            self._sleep()

    def _parse_card(self, card) -> dict | None:
        title_tag = card.select_one("h2, h3, .title, .course-title")
        if not title_tag:
            return None

        title = title_tag.get_text(strip=True)

        link_tag = card.find("a", href=True)
        link = urljoin(TUTORTOP_BASE, link_tag["href"]) if link_tag else None

        full_text = card.get_text(" ", strip=True)
        price = self._extract_price(full_text)
        duration_months = self._extract_first(MONTHS_RE, full_text)
        duration_hours = self._extract_first(HOURS_RE, full_text)

        school_tag = card.select_one(".school, .author, .provider")
        school = school_tag.get_text(strip=True) if school_tag else None

        fmt = self._guess_format(full_text)

        return {
            "title": title,
            "school": school,
            "url": link,
            "price_rub": price,
            "duration_months": duration_months,
            "duration_hours": duration_hours,
            "format": fmt,
        }

    @staticmethod
    def _extract_price(text: str) -> float | None:
        match = PRICE_RE.search(text)
        if not match:
            return None
        digits = re.sub(r"\s+", "", match.group(1))
        try:
            return float(digits)
        except ValueError:
            return None

    @staticmethod
    def _extract_first(pattern: re.Pattern, text: str) -> int | None:
        match = pattern.search(text)
        return int(match.group(1)) if match else None

    @staticmethod
    def _guess_format(text: str) -> str:
        text_low = text.lower()
        if "офлайн" in text_low or "очно" in text_low:
            return "offline"
        if "вебинар" in text_low or "поток" in text_low:
            return "online_group"
        if "индивидуал" in text_low or "ментор" in text_low or "наставник" in text_low:
            return "online_mentor"
        return "online_self"