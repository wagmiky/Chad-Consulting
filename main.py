"""Точка входа в проект.

Пайплайн стандартный: собрать сырьё из трёх источников, слепить из них
единый набор конкурентов, прогнать анализ и сохранить отчёт. Чтобы не
дергать сеть на каждом запуске, коллекторы кешируют результат в CSV.

Запуск:
    python main.py --full          собрать данные и сделать отчёт
    python main.py --only collect  только сбор
    python main.py --only analyze  только анализ из кеша
    python main.py --refresh       принудительно обновить кеш
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from analysis import CompetitorRegistry, PositioningReport
from collectors import TutortopCollector, YouTubeCollector, VKGroupsCollector
from config import (
    CLEAN_COMPETITORS_CSV,
    RAW_COURSES_CSV,
    RAW_VK_CSV,
    RAW_YOUTUBE_CSV,
    REPORTS_DIR,
)
from visualization.plots import PlotBuilder


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def collect(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    courses = TutortopCollector(RAW_COURSES_CSV, refresh=refresh).collect()
    youtube = YouTubeCollector(RAW_YOUTUBE_CSV, refresh=refresh).collect()
    vk = VKGroupsCollector(RAW_VK_CSV, refresh=refresh).collect()
    return courses, youtube, vk


def build_registry(courses, youtube, vk) -> pd.DataFrame:
    registry = CompetitorRegistry.from_frames(courses, youtube, vk)
    frame = registry.to_frame()
    frame.to_csv(CLEAN_COMPETITORS_CSV, index=False, encoding="utf-8-sig")
    return frame


def analyze(frame: pd.DataFrame) -> None:
    report = PositioningReport(frame, our_price_target=9900)
    md_path = report.render_markdown(REPORTS_DIR / "positioning.md")
    print(f"Отчёт сохранён: {md_path}")

    plots = PlotBuilder(frame)
    fig = plots.price_distribution(save_to=REPORTS_DIR / "price_distribution.png")
    if fig is not None:
        print("Сохранён график цен по форматам: reports/price_distribution.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Собрать и проанализировать")
    parser.add_argument("--only", choices=["collect", "analyze"], help="Этап пайплайна")
    parser.add_argument("--refresh", action="store_true", help="Игнорировать кеш")
    args = parser.parse_args()

    setup_logging()

    if args.only == "collect" or args.full:
        courses, youtube, vk = collect(refresh=args.refresh)
        frame = build_registry(courses, youtube, vk)
        print(f"Сохранено конкурентов: {len(frame)}")

    if args.only == "analyze" or args.full:
        if not Path(CLEAN_COMPETITORS_CSV).exists():
            raise SystemExit("Сначала запустите сбор: python main.py --only collect")
        frame = pd.read_csv(CLEAN_COMPETITORS_CSV)
        analyze(frame)

    if not args.full and not args.only:
        parser.print_help()


if __name__ == "__main__":
    main()