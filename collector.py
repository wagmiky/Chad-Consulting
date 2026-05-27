"""
Сборщики данных по конкурентам.

Два источника:
1) HTML-снимок статьи vc.ru с подборкой онлайн-курсов для стилистов-имиджмейкеров.
   Парсим BeautifulSoup-ом, регулярками вытаскиваем цены и длительности.
2) YouTube-каналы блогеров по теме (yt-dlp, либо локальный кеш).

Запуск парсинга с обновлением:
    python run.py --refresh-courses   скачать свежую статью с vc.ru
    python run.py --refresh           заодно обновить YouTube-кеш через yt-dlp
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


DATA_DIR = Path(__file__).parent / "data"

# Источники для парсинга курсов. Это две публичные статьи-подборки на vc.ru,
# регулярно обновляются редакцией. У них одинаковая разметка карточек
# курсов (заголовок школы + список "Стоимость", "Продолжительность",
# "Формат"), поэтому один и тот же парсер работает на обоих.
VCRU_URLS = [
    # стиль и имидж
    "https://vc.ru/edu/580574-25-onlain-kursov-dlya-stilistov-imidzhmeikerov-platnye-i-besplatnye",
    "https://vc.ru/education/2667003-luchshie-kursy-dlya-stilistov-imidzhmeykerov",
    # уход и косметология
    "https://vc.ru/edu/814917-onlain-obuchenie-kosmetologii-14-onlain-kursov-6-besplatnyh",
    "https://vc.ru/edu/788464-top-30-kursov-po-massazhu-lica-obuchenie-v-moskve-i-onlain",
    # фитнес-тренеры (работа над телосложением)
    "https://vc.ru/s/2196147-vskurses/793376-top-15-luchshih-kursov-fitnes-trenera-reyting-obucheniya-2023",
]

# вежливый User-Agent, без него vc.ru может вернуть 403
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15"
)


class CourseScraper:
    """Парсер карточек курсов со снимка статьи vc.ru.

    Логика:
      1) если включён refresh или локального HTML нет, идём в сеть и
         сохраняем снимок в data/courses_catalog.html;
      2) парсим сохранённый HTML, ищем заголовки школ и блоки параметров;
      3) для каждого курса достаём цену, длительность и формат.
    """

    # ключевые слова, по которым раскладываем формат
    format_map = [
        ("живые вебинары", "online_group"),
        ("живые лекции", "online_group"),
        ("онлайн-лекции", "online_group"),
        ("онлайн-вебинар", "online_group"),
        ("групповые занятия", "online_group"),
        ("куратор", "online_mentor"),
        ("ментор", "online_mentor"),
        ("наставник", "online_mentor"),
        ("преподаватель онлайн", "online_mentor"),
        ("консультации", "online_mentor"),
        ("стажировка", "online_mentor"),
        ("workshop", "online_mentor"),
        ("очно", "offline"),
        ("в москве", "offline"),
        ("видеолекции", "online_self"),
        ("видеоуроки", "online_self"),
    ]

    def __init__(self, urls=None, cache_dir=None):
        self.urls = urls if urls is not None else VCRU_URLS
        self.cache_dir = Path(cache_dir) if cache_dir else DATA_DIR

    def load(self, refresh=False):
        all_rows = []
        for i, url in enumerate(self.urls, start=1):
            cache_path = self.cache_dir / f"courses_catalog_{i}.html"
            if refresh or not cache_path.exists():
                ok = self._download(url, cache_path)
                if not ok:
                    print(f"  источник {i}: не скачался, пропускаем")
                    continue
            if not cache_path.exists():
                print(f"  источник {i}: HTML отсутствует, пропускаем")
                continue
            html = cache_path.read_text(encoding="utf-8")
            rows = self._extract_courses(html)
            print(f"  источник {i}: {len(rows)} курсов")
            for r in rows:
                r["source_url"] = url
            all_rows.extend(rows)

        df = pd.DataFrame(all_rows)
        if df.empty:
            return df

        df = df.drop_duplicates(subset=["school", "title"], keep="first").reset_index(drop=True)
        df["source"] = "vc.ru"
        return df

    @staticmethod
    def _download(url, cache_path):
        print(f"  скачиваю {url}")
        cache_path.parent.mkdir(exist_ok=True)
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            r.raise_for_status()
        except requests.exceptions.RequestException as exc:
            print(f"    ошибка: {exc}")
            return False
        cache_path.write_text(r.text, encoding="utf-8")
        return True

    # форматы разметки школы в заголовке курса на vc.ru
    SCHOOL_PATTERNS = [
        re.compile(r"^»\s*[—–\-]\s+(.+)$"),    # » — Talentsy
        re.compile(r"^»\s*от\s+(.+)$"),        # » от Bantser Fashion School
        re.compile(r"^[—–\-]\s+(.+)$"),        # — Fashion Factory
        re.compile(r"^от\s+([A-ZА-ЯЁ].+)$"),   # от Fashion Factory
    ]

    def _extract_courses(self, html):
        """
        Парсим статью vc.ru, отталкиваясь от строк со "Стоимость:".

        Алгоритм:
          1. находим все строки "Стоимость: N ₽";
          2. для каждой ищем ближайшую сверху строку с разметкой школы
             (одного из трёх форматов: "— Школа", "» от Школа" или "от Школа");
          3. между предыдущей "Стоимость" и найденной школой лежит название курса;
          4. от "Стоимость:" вперёд до следующей "Стоимость:" — параметры курса.
        """
        soup = BeautifulSoup(html, "lxml")
        article = soup.find("article") or soup.find("main") or soup
        text = article.get_text("\n", strip=True)
        lines = text.split("\n")

        price_lines = [
            i for i, ln in enumerate(lines)
            if re.match(r"^Стоимость[:\s]", ln.strip())
        ]

        rows = []
        for k, pi in enumerate(price_lines):
            prev_pi = price_lines[k - 1] if k > 0 else -1
            search_from = max(prev_pi + 1, 0)

            # ищем СНИЗУ ВВЕРХ ближайшую разметку школы (ближайшая к Стоимости важнее)
            school = None
            school_idx = None
            for j in range(pi - 1, search_from - 1, -1):
                ln = lines[j].strip()
                if not ln:
                    continue
                for pat in self.SCHOOL_PATTERNS:
                    m2 = pat.match(ln)
                    if m2:
                        candidate = m2.group(1).strip(" «»\"'")
                        if 2 <= len(candidate) <= 80 and not self._looks_like_description(candidate):
                            school = candidate
                            school_idx = j
                            break
                if school:
                    break

            if not school:
                continue

            # название: 1-3 строки СРАЗУ ПЕРЕД школой. Идём вверх от school_idx-1,
            # собираем подряд "информативные" строки. Останавливаемся как только
            # встречаем номер курса, "→", описание или хвост предыдущего курса.
            title_parts = []
            for j in range(school_idx - 1, max(school_idx - 6, search_from - 1), -1):
                ln = lines[j].strip()
                if not ln:
                    if title_parts:
                        break
                    continue
                # одиночные кавычки пропускаем
                if ln in ("«", "»", "\"", "'"):
                    continue
                # номер курса = граница, не идём дальше
                if re.match(r"^\d{1,2}\.\s*$", ln):
                    break
                # строка вида "N. что-то" = начало курса, забираем содержимое и стоп
                m_num = re.match(r"^\d{1,2}\.\s*[«\"']?(.+)", ln)
                if m_num:
                    candidate = m_num.group(1).strip(" «»\"'")
                    if candidate and not self._looks_like_description(candidate):
                        title_parts.insert(0, candidate)
                    break
                # хвост предыдущего курса
                if "Посмотреть программу" in ln or ln.startswith("→"):
                    break
                if self._looks_like_description(ln):
                    break
                title_parts.insert(0, ln)

            if not title_parts:
                continue

            title = " ".join(title_parts).strip(" «»\"'")
            if len(title) > 200:
                title = title[:200]

            next_pi = price_lines[k + 1] if k + 1 < len(price_lines) else len(lines)
            chunk = "\n".join(lines[pi:next_pi])

            price = self._extract_price(chunk)
            months = self._extract_months(chunk)
            if price is None or months is None:
                continue

            fmt_raw = self._extract_format_text(chunk)
            fmt = self.normalize_format(fmt_raw)

            rows.append({
                "title": title,
                "school": school,
                "price_rub": price,
                "duration_months": months,
                "format_raw": fmt_raw[:200],
                "format": fmt,
            })

        return rows

    @staticmethod
    def _looks_like_description(text):
        """Эвристика: длинные предложения с глаголами это не заголовок."""
        if len(text) > 90:
            return True
        verbs = (" учит", "включ", "состоит", "обучен", "позвол", "помога",
                 "пройти", "получи", "освои", "уроков", "видео")
        low = text.lower()
        return any(w in low for w in verbs)

    @staticmethod
    def _extract_price(chunk):
        # ищем первую строку вида "Стоимость: ... ₽" в первых 800 символах
        # (после идут уже параметры рассрочки и прочее)
        head = chunk[:1500]
        m = re.search(r"Стоимость[:\s]+(?:от\s*)?([\d\s]+)\s*₽", head)
        if not m:
            return None
        digits = re.sub(r"\D", "", m.group(1))
        return int(digits) if digits else None

    @staticmethod
    def _extract_months(chunk):
        """
        Понимает варианты:
          "3 месяца", "6 месяцев", "2,5 месяца"
          "от 1 до 3 месяцев" — берём верхнюю границу
          "1,5 года", "8 недель", "4 часа 50 минут"
        Берём последнюю пару "число + единица длительности" в строке.
        """
        head = chunk[:1500]
        m = re.search(r"Продолжительность[^\n]*", head)
        if not m:
            return None
        line = m.group(0)
        matches = re.findall(
            r"(\d+(?:[,.]\d+)?)\s*"
            r"(месяц[аев]*|год[а]?|лет|недел[яьи]+|час[аов]*)",
            line,
        )
        if not matches:
            return None
        value_str, unit = matches[-1]
        value = float(value_str.replace(",", "."))
        unit = unit.lower()
        if unit.startswith("месяц"):
            return int(round(value))
        if unit.startswith("год") or unit == "лет":
            return int(round(value * 12))
        if unit.startswith("недел"):
            return max(1, int(round(value / 4)))
        if unit.startswith("час"):
            return 1
        return None

    @staticmethod
    def _extract_format_text(chunk):
        head = chunk[:2000]
        m = re.search(r"Формат[:\s]+([^\n]+)", head)
        return m.group(1) if m else ""

    @classmethod
    def normalize_format(cls, text):
        low = text.lower()
        for word, code in cls.format_map:
            if word in low:
                return code
        return "online_self"


class YouTubeFetcher:
    """Берёт метаданные YouTube-каналов через yt-dlp.

    При первом вызове из сети читает yt-dlp данные, кеширует в CSV.
    Дальше работает с кешем, если не передан refresh=True.
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


class TelegramScraper:
    """Парсер публичных Telegram-каналов через web-preview t.me/s/CHANNEL.

    Достаёт:
      - название канала;
      - число подписчиков (если есть на странице);
      - короткое описание;
      - количество последних видимых постов.

    Не использует Telegram API, парсит публичную страницу, поэтому работает
    без токенов. Если канал приватный, ничего не выдаст.
    """

    def __init__(self, channels, cache_path=None):
        self.channels = channels
        self.cache_path = Path(cache_path) if cache_path else DATA_DIR / "telegram_cache.csv"

    def load(self, refresh=False):
        if not refresh and self.cache_path.exists():
            return pd.read_csv(self.cache_path)

        rows = []
        for ch in self.channels:
            try:
                rows.append(self._fetch_one(ch))
            except Exception as exc:
                print(f"  TG {ch} не отработал: {exc}")
        if not rows:
            return pd.DataFrame(
                columns=["channel", "url", "subscribers", "posts", "description", "source"]
            )
        df = pd.DataFrame(rows)
        df["source"] = "telegram"
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(self.cache_path, index=False, encoding="utf-8-sig")
        return df

    def _fetch_one(self, channel):
        url = f"https://t.me/s/{channel.lstrip('@')}"
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        title_el = soup.find("div", class_="tgme_channel_info_header_title")
        title = title_el.get_text(strip=True) if title_el else channel

        desc_el = soup.find("div", class_="tgme_channel_info_description")
        description = desc_el.get_text(" ", strip=True)[:200] if desc_el else ""

        # счётчики: подписчики, фото, видео, посты
        subscribers = None
        posts = None
        for counter in soup.find_all("div", class_="tgme_channel_info_counter"):
            value_el = counter.find("span", class_="counter_value")
            type_el = counter.find("span", class_="counter_type")
            if not value_el or not type_el:
                continue
            value_text = value_el.get_text(strip=True)
            label = type_el.get_text(strip=True).lower()
            value = self._parse_count(value_text)
            if "подписчик" in label or "subscriber" in label or "member" in label:
                subscribers = value
            elif "постов" in label or "посты" in label or "пост" in label or "post" in label:
                posts = value

        return {
            "channel": title,
            "url": url,
            "subscribers": subscribers,
            "posts": posts,
            "description": description,
        }

    @staticmethod
    def _parse_count(text):
        """Понимает '12.3K', '1.5M', '8 155' и т.п."""
        text = text.replace(" ", "").replace("\xa0", "")
        m = re.match(r"([\d.,]+)([KMkКМмk]?)", text)
        if not m:
            return None
        try:
            value = float(m.group(1).replace(",", "."))
        except ValueError:
            return None
        unit = m.group(2).lower()
        if unit in ("k", "к"):
            value *= 1_000
        elif unit in ("m", "м"):
            value *= 1_000_000
        return int(value)


def merge_competitors(courses, youtube, telegram=None):
    """
    Сводит школы, YouTube и Telegram в одну таблицу конкурентов.
    """
    schools = courses.rename(columns={"school": "name"})[
        ["name", "title", "price_rub", "duration_months", "format", "source"]
    ].copy()

    bloggers = youtube.rename(columns={"channel": "name", "subscribers": "audience"})[
        ["name", "url", "audience", "avg_views", "source"]
    ].copy()
    bloggers["format"] = "personal_brand"

    parts = [schools, bloggers]
    if telegram is not None and not telegram.empty:
        tg = telegram.rename(columns={"channel": "name", "subscribers": "audience"})[
            ["name", "url", "audience", "source"]
        ].copy()
        tg["format"] = "telegram_community"
        parts.append(tg)

    return pd.concat(parts, ignore_index=True)
