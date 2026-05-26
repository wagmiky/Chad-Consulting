"""
Точка входа в проект. Собирает данные из двух источников, считает
анализ, рисует графики, пишет markdown отчёт.

    python run.py            собрать всё, посчитать, сохранить отчёт
    python run.py --refresh  обновить YouTube кеш через yt-dlp
"""

import argparse

from collector import CourseScraper, YouTubeFetcher, merge_competitors
from analysis import CompetitorAnalysis, REPORTS_DIR


# каналы, которые тянем через yt-dlp при --refresh
YT_CHANNELS = [
    "https://www.youtube.com/@menslook",
    "https://www.youtube.com/@stilbezshuma",
    "https://www.youtube.com/@kosmpacka",
    "https://www.youtube.com/@lookslab",
    "https://www.youtube.com/@groomking",
    "https://www.youtube.com/@imageman",
    "https://www.youtube.com/@skinbeard",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="заново сходить в YouTube")
    args = p.parse_args()

    print("Парсим HTML каталог курсов...")
    courses = CourseScraper().load()
    print(f"  нашли {len(courses)} карточек")

    print("Берём данные по YouTube каналам...")
    youtube = YouTubeFetcher(YT_CHANNELS).load(refresh=args.refresh)
    print(f"  {len(youtube)} каналов")

    merged = merge_competitors(courses, youtube)
    merged.to_csv(REPORTS_DIR / "competitors.csv", index=False, encoding="utf-8-sig")

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
