# Анализ конкурентов: онлайн-курсы по внешнему виду

Часть проекта Chad Consulting. Здесь разбираем конкурентов, чтобы понять
ценовое позиционирование и форматы школ по стилю, уходу за кожей и
луксмаксингу для парней 18-28 лет.

## Источники данных

1. **HTML-каталог курсов** (`data/courses_catalog.html`). Это снапшот
   подборки предложений по теме, собранной из публичных страниц онлайн-школ.
   HTML сохранён один раз, дальше работа идёт офлайн через BeautifulSoup.
   Так делают в проде, чтобы не дёргать сайты лишний раз и иметь
   воспроизводимый набор данных. Из карточек достаются школа, цена,
   длительность и формат.
2. **YouTube-каналы блогеров** по теме, через `yt-dlp`. Берём число
   подписчиков и средние просмотры. Для стабильности данные закешированы
   в `data/youtube_cache.csv`, обновляются через `python run.py --refresh`.

## Что считается

- Базовая статистика по ценам в разрезе формата (median, mean, min, max).
- Две гипотезы через Mann-Whitney U (one-sided):
  - «курсы с ментором дороже самостоятельных»;
  - «офлайн дороже онлайна в пересчёте на месяц обучения».
- Spearman-корреляция цены и длительности.
- Доверительный интервал медианы цены через bootstrap (2000 итераций).
- Elbow plot для выбора k и кластеризация конкурентов (k-means по цене и
  длительности), даёт автоматические «портреты» сегментов.
- Метрики по YouTube-конкурентам: суммарная и медианная аудитория.

## Запуск

```bash
pip install -r requirements.txt
python run.py              # сбор, анализ, отчёт
python run.py --refresh    # пересобрать YouTube кеш через yt-dlp
python dashboard.py        # интерактивный дашборд на :8050
```

После `run.py` появляются файлы:
- `reports/report.md`: итоговый отчёт
- `reports/price_boxplot.png`
- `reports/price_vs_duration.png`
- `reports/elbow.png`
- `reports/clusters.png`
- `reports/youtube_audience.png`
- `reports/competitors.csv`: объединённая таблица конкурентов

## Структура

```
final_project/
├── data/
│   ├── courses_catalog.html   снапшот HTML каталога
│   └── youtube_cache.csv      кеш по YouTube
├── collector.py               CourseScraper, YouTubeFetcher, merge_competitors
├── analysis.py                CompetitorAnalysis: метрики, гипотезы, графики, отчёт
├── dashboard.py               Dash приложение
├── run.py                     точка входа CLI
└── reports/                   результаты
```

## Что закрыто из тем курса

- **ООП**: три класса с разной зоной ответственности (`CourseScraper`,
  `YouTubeFetcher`, `CompetitorAnalysis`).
- **Dash дашборд**: `dashboard.py` с тремя вкладками.
- **Гипотезы и оценка выборок**: Mann-Whitney + bootstrap CI + Spearman.
- **Графики**: matplotlib для статики и Plotly внутри дашборда.
- **Парсинг**: BeautifulSoup на сохранённой HTML странице.
- **Кластеризация**: k-means с выбором k по elbow.
