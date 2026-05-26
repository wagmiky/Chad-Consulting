# Анализ конкурентов — онлайн-курсы по внешнему виду

Часть проекта Chad Consulting. Тут разбираю конкурентов, чтобы понять
ценовое позиционирование и форматы, на которых строятся школы по стилю,
уходу за кожей и луксмаксингу.

## Источники данных

1. **HTML-каталог курсов** — статический парсинг через BeautifulSoup
   (`data/courses_catalog.html`). Из карточек достаются школа, цена,
   длительность и формат.
2. **YouTube-каналы блогеров** по теме — через `yt-dlp`. Берём число
   подписчиков и средние просмотры. Для скорости/устойчивости данные
   закешированы в `data/youtube_cache.csv`.

## Что считается

- Базовая статистика по ценам в разрезе формата (median, mean, min/max).
- Две гипотезы через Mann-Whitney U (one-sided):
  - «курсы с ментором дороже самостоятельных»
  - «длинные курсы (≥ 3 мес) дороже коротких»
- Spearman-корреляция между ценой и длительностью.
- Доверительный интервал медианы цены через bootstrap.
- Кластеризация конкурентов (k-means по цене и длительности, k=3) —
  даёт автоматические «портреты» сегментов.
- Метрики по YouTube-конкурентам: суммарная и медианная аудитория.

## Запуск

```bash
pip install -r requirements.txt
python run.py              # сбор + анализ + отчёт
python run.py --refresh    # пересобрать YouTube кеш через yt-dlp
python dashboard.py        # интерактивный дашборд на :8050
```

После `run.py` появляются файлы:
- `reports/report.md` — итоговый отчёт
- `reports/price_boxplot.png`
- `reports/price_vs_duration.png`
- `reports/clusters.png`
- `reports/youtube_audience.png`
- `reports/competitors.csv` — объединённая таблица конкурентов

## Структура

```
chad-seva/
├── data/
│   ├── courses_catalog.html   — сохранённый HTML каталога
│   └── youtube_cache.csv      — кеш по YouTube
├── collector.py               — CourseScraper и YouTubeFetcher
├── analysis.py                — CompetitorAnalysis: метрики, гипотеза, графики, отчёт
├── dashboard.py               — Dash приложение
├── run.py                     — точка входа CLI
└── reports/                   — куда складываются результаты
```

## Что закрыто из тем курса

- **ООП** — три класса: `CourseScraper`, `YouTubeFetcher`, `CompetitorAnalysis`.
- **Dash дашборд** — `dashboard.py` с двумя вкладками.
- **Гипотезы и оценка выборок** — Mann-Whitney + bootstrap CI.
- **Графики** — matplotlib (статика) + Plotly (внутри дашборда).
- **Парсинг** — BeautifulSoup на сохранённой HTML странице.
