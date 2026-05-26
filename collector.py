"""
Сборщики данных по конкурентам.

Два источника:
1) HTML-каталог курсов, статический парсинг через BeautifulSoup
2) YouTube-каналы блогеров по теме (yt-dlp, либо локальный кеш)
"""

import re
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


DATA_DIR = Path(__file__).parent / "data"


class CourseScraper:
    """Парсер карточек курсов из сохранённой HTML страницы каталога.

    Алгоритм простой: открыли HTML, нашли все li.course-card, из каждой
    карточки вытащили заголовок, школу, цену, длительность и формат.
    Если в карточке какого-то поля нет (битая разметка), она пропускается.
    """

    # ключевые слова в описании формата и куда мы их маппим
    format_map = [
        ("куратор", "online_mentor"),
        ("ментор", "online_mentor"),
        ("наставник", "online_mentor"),
        ("вебинар", "online_group"),
        ("поток", "online_group"),
        ("офлайн", "offline"),
        ("в зале", "offline"),
    ]

    def __init__(self, html_path=None):
        self.html_path = Path(html_path) if html_path else DATA_DIR / "courses_catalog.html"

    def load(self):
        html = self.html_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")

        rows = []
        skipped = 0
        for card in soup.select("li.course-card"):
            parsed = self.parse_card(card)
            if parsed is None:
                skipped += 1
                continue
            rows.append(parsed)

        if skipped:
            print(f"  пропущено карточек с битой разметкой: {skipped}")

        df = pd.DataFrame(rows)
        df["source"] = "catalog"
        return df

    def parse_card(self, card):
        """Возвращает dict с полями курса, либо None если карточка битая."""
        try:
            title = card.select_one(".title").get_text(strip=True)
            school = card.select_one(".school").get_text(strip=True)
            price_text = card.select_one(".price").get_text(strip=True)
            duration_text = card.select_one(".duration").get_text(strip=True)
            format_text = card.select_one(".format").get_text(strip=True)
        except AttributeError:
            return None

        return {
            "title": title,
            "school": school,
            "price_rub": self.parse_price(price_text),
            "duration_months": self.parse_months(duration_text),
            "format_raw": format_text,
            "format": self.normalize_format(format_text),
        }

    @staticmethod
    def parse_price(text):
        digits = re.sub(r"\D", "", text)
        return int(digits) if digits else None

    @staticmethod
    def parse_months(text):
        m = re.search(r"(\d+)", text)
        return int(m.group(1)) if m else None

    @classmethod
    def normalize_format(cls, text):
        low = text.lower()
        for word, code in cls.format_map:
            if word in low:
                return code
        # если ни одно ключевое слово не сработало, считаем что это онлайн в записи
        return "online_self"


class YouTubeFetcher:
    """Берёт метаданные YouTube-каналов через yt-dlp.

    Если интернета нет или yt-dlp падает, читаем заранее собранный
    кеш в CSV. Так демо не зависит от сети.
    """

    def __init__(self, channels, cache_path=None):
        self.channels = channels
        self.cache_path = Path(cache_path) if cache_path else DATA_DIR / "youtube_cache.csv"

    def load(self, refresh=False):
        if not refresh and self.cache_path.exists():
            return pd.read_csv(self.cache_path)

        try:
            rows = list(self._fetch_live())
        except Exception as exc:
            print(f"yt-dlp не отработал ({exc}), беру кеш")
            return pd.read_csv(self.cache_path)

        df = pd.DataFrame(rows)
        df["source"] = "youtube"
        df.to_csv(self.cache_path, index=False, encoding="utf-8-sig")
        return df

    def _fetch_live(self):
        # импорт внутри функции: чтобы без yt-dlp всё равно можно было читать кеш
        from yt_dlp import YoutubeDL

        opts = {
            "quiet": True,
            "skip_download": True,
            "extract_flat": "in_playlist",
            "playlistend": 20,
            "no_warnings": True,
        }
        with YoutubeDL(opts) as ydl:
            for url in self.channels:
                info = ydl.extract_info(url + "/videos", download=False)
                entries = info.get("entries", []) or []
                views = [e.get("view_count") or 0 for e in entries]
                avg = int(sum(views) / len(views)) if views else 0
                yield {
                    "channel": info.get("channel") or info.get("uploader"),
                    "url": url,
                    "subscribers": info.get("channel_follower_count"),
                    "avg_views": avg,
                    "videos_sampled": len(entries),
                }
                time.sleep(0.5)


def merge_competitors(courses, youtube):
    """
    Сводит школы и блогеров в одну таблицу конкурентов.
    Школы дают цены и форматы, блогеры показывают охват аудитории.
    """
    schools = courses.rename(columns={"school": "name"})[
        ["name", "title", "price_rub", "duration_months", "format", "source"]
    ].copy()

    bloggers = youtube.rename(columns={"channel": "name", "subscribers": "audience"})[
        ["name", "url", "audience", "avg_views", "source"]
    ].copy()
    bloggers["format"] = "personal_brand"

    merged = pd.concat([schools, bloggers], ignore_index=True)
    return merged
