"""
Точка входа в проект. Собирает данные из трёх источников, считает
анализ, рисует графики, пишет markdown отчёт.

    python run.py                       посчитать на текущем кеше
    python run.py --refresh-courses     скачать свежие снимки vc.ru
    python run.py --refresh             обновить YouTube кеш через yt-dlp
    python run.py --refresh-telegram    обновить TG кеш с t.me/s/...
    python run.py --refresh-all         обновить всё разом
"""

import argparse

from collector import (
    CourseScraper, YouTubeFetcher, TelegramScraper, merge_competitors,
)
from analysis import CompetitorAnalysis, REPORTS_DIR


# YouTube каналы по теме внешнего вида для парней
YT_CHANNELS = [
    "https://www.youtube.com/@ROGOVLIVE",
    "https://www.youtube.com/@SHLYAPA_pronin",
    "https://www.youtube.com/@egor_nastoyashiy",
    "https://www.youtube.com/@feministrategy",
    "https://www.youtube.com/@SalavatKypere",
]

# публичные Telegram-каналы по луксмаксингу, мужским причёскам и уходу.
# Проверены на существование через TGStat и личные референсы.
TG_CHANNELS = [
    "Menshairstyling",
    "son_godd",
    "Looksmax",
    "kovbridg",
    "menshairstyles2020",
    "menslooks",
    "haircutsboyz",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refresh-courses", action="store_true",
                   help="скачать свежие снимки статей vc.ru")
    p.add_argument("--refresh", action="store_true",
                   help="обновить YouTube кеш через yt-dlp")
    p.add_argument("--refresh-telegram", action="store_true",
                   help="обновить Telegram кеш через t.me/s/")
    p.add_argument("--refresh-all", action="store_true",
                   help="обновить все три источника")
    args = p.parse_args()

    if args.refresh_all:
        args.refresh_courses = True
        args.refresh = True
        args.refresh_telegram = True

    print("Парсим HTML каталоги курсов с vc.ru...")
    courses = CourseScraper().load(refresh=args.refresh_courses)
    print(f"  курсов с ценами: {len(courses)}")

    print("Берём данные по YouTube каналам...")
    youtube = YouTubeFetcher(YT_CHANNELS).load(refresh=args.refresh)
    print(f"  каналов YouTube: {len(youtube)}")

    print("Берём данные по Telegram каналам...")
    telegram = TelegramScraper(TG_CHANNELS).load(refresh=args.refresh_telegram)
    print(f"  каналов Telegram: {len(telegram)}")

    merged = merge_competitors(courses, youtube, telegram)
    merged.to_csv(REPORTS_DIR / "competitors.csv", index=False, encoding="utf-8-sig")

    if len(courses) == 0:
        print("\nКурсов нет, гипотезы по ценам не считаем.")
        print("Дашборд и YouTube-аналитику всё равно собираем.")
        return

    print("Считаем анализ и пишем отчёт...")
    analysis = CompetitorAnalysis(courses, youtube)

    analysis.plot_price_boxplot(save_to=REPORTS_DIR / "price_boxplot.png")
    analysis.plot_price_vs_duration(save_to=REPORTS_DIR / "price_vs_duration.png")
    analysis.plot_elbow(save_to=REPORTS_DIR / "elbow.png")
    analysis.plot_clusters(save_to=REPORTS_DIR / "clusters.png")
    analysis.plot_youtube_audience(save_to=REPORTS_DIR / "youtube_audience.png")

    report_path = analysis.render_markdown(REPORTS_DIR / "report.md")
    print(f"\nГотово. Отчёт: {report_path}")
    print(f"Графики и таблица: папка {REPORTS_DIR.name}/")
    print("\nДля интерактивного дашборда: python dashboard.py")


if __name__ == "__main__":
    main()
