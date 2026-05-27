"""
Интерактивный дашборд на Dash. Три вкладки: обзор по школам,
карта позиционирования и кластеры с гипотезами.

Запуск: python dashboard.py
"""

import plotly.express as px
from dash import Dash, dcc, html, dash_table, Input, Output

from collector import CourseScraper, YouTubeFetcher, TelegramScraper
from analysis import CompetitorAnalysis


def _prepare():
    courses = CourseScraper().load()
    youtube = YouTubeFetcher(channels=[]).load()        # из кеша
    telegram = TelegramScraper(channels=[]).load()      # из кеша
    analysis = CompetitorAnalysis(courses, youtube)
    return courses, youtube, telegram, analysis


def make_app():
    courses, youtube, telegram, analysis = _prepare()

    fmt_options = [{"label": "Все форматы", "value": "all"}]
    for f in sorted(courses["format"].dropna().unique()):
        fmt_options.append({"label": f, "value": f})

    h1 = analysis.test_target_below_market()
    corr = analysis.price_duration_correlation()
    clusters = analysis.cluster_competitors()

    app = Dash(__name__, title="Конкуренты: внешний вид")

    app.layout = html.Div(
        style={"fontFamily": "system-ui, sans-serif", "padding": "20px"},
        children=[
            html.H1("Анализ конкурентов: курсы по стилю и уходу"),
            html.P("Данные: HTML-каталог курсов и YouTube-каналы блогеров."),

            dcc.Tabs([

                dcc.Tab(label="Обзор школ", children=[
                    html.Div(style={"marginTop": "12px"}, children=[
                        html.Label("Формат:"),
                        dcc.Dropdown(
                            id="fmt-filter",
                            options=fmt_options,
                            value="all",
                            clearable=False,
                            style={"width": "260px"},
                        ),
                    ]),
                    dash_table.DataTable(
                        id="courses-table",
                        columns=[
                            {"name": c, "id": c}
                            for c in ["title", "school", "format", "price_rub", "duration_months"]
                        ],
                        page_size=10,
                        style_table={"overflowX": "auto", "marginTop": "12px"},
                        style_cell={"fontSize": "13px", "padding": "6px"},
                    ),
                    dcc.Graph(id="boxplot"),
                ]),

                dcc.Tab(label="Карта позиционирования", children=[
                    dcc.Graph(
                        figure=px.scatter(
                            courses.dropna(subset=["price_rub", "duration_months"]),
                            x="duration_months",
                            y="price_rub",
                            color="format",
                            hover_data=["title", "school"],
                            title="Курсы: цена и длительность",
                            template="simple_white",
                            labels={"duration_months": "Длительность, мес",
                                    "price_rub": "Цена, руб"},
                        )
                    ),
                ]),

                dcc.Tab(label="Социальные конкуренты", children=_social_tab(youtube, telegram)),

                dcc.Tab(label="Кластеры и гипотезы", children=_clusters_and_hypotheses_tab(
                    clusters, h1, corr
                )),
            ]),
        ],
    )

    @app.callback(
        Output("courses-table", "data"),
        Output("boxplot", "figure"),
        Input("fmt-filter", "value"),
    )
    def filter_data(value):
        view = courses if value == "all" else courses[courses["format"] == value]
        fig = px.box(
            view.dropna(subset=["price_rub"]),
            x="format",
            y="price_rub",
            color="format",
            title="Цены по форматам",
            template="simple_white",
        )
        return view.to_dict("records"), fig

    return app


def _social_tab(youtube, telegram):
    children = [html.H4("Бесплатный контент: прямые конкуренты нашего продукта")]

    if youtube is not None and not youtube.empty:
        yt = youtube.dropna(subset=["subscribers"]).sort_values("subscribers", ascending=True)
        children.append(dcc.Graph(figure=px.bar(
            yt,
            x="subscribers", y="channel", orientation="h",
            title="YouTube-каналы по числу подписчиков",
            template="simple_white",
        ).update_layout(yaxis={"categoryorder": "total ascending"})))

    if telegram is not None and not telegram.empty:
        tg = telegram.dropna(subset=["subscribers"]).sort_values("subscribers", ascending=True)
        if not tg.empty:
            children.append(dcc.Graph(figure=px.bar(
                tg,
                x="subscribers", y="channel", orientation="h",
                title="Telegram-каналы по числу подписчиков",
                template="simple_white",
            ).update_layout(yaxis={"categoryorder": "total ascending"})))

        total_yt = int(youtube["subscribers"].dropna().sum()) if youtube is not None and not youtube.empty else 0
        total_tg = int(telegram["subscribers"].dropna().sum())
        children.append(html.P(
            f"Суммарная аудитория: YouTube {total_yt:,}, Telegram {total_tg:,}".replace(",", " ")
        ))

    return children


def _clusters_and_hypotheses_tab(clusters, h1, corr):
    children = []
    if clusters is not None:
        cl = clusters["data"]
        ports = clusters["portraits"]
        cl = cl.merge(
            ports["portrait"].rename("cluster_label"),
            left_on="cluster",
            right_index=True,
        )
        fig = px.scatter(
            cl,
            x="duration_months",
            y="price_rub",
            color="cluster_label",
            hover_data=["title", "school", "format"],
            title="Кластеры конкурентов (k-means, k=3)",
            template="simple_white",
            labels={"duration_months": "Длительность, мес",
                    "price_rub": "Цена, руб",
                    "cluster_label": "Портрет"},
        )
        children.append(dcc.Graph(figure=fig))

        ports_view = ports.reset_index().assign(
            avg_price=lambda d: d["avg_price"].astype(int),
            avg_duration=lambda d: d["avg_duration"].astype(int),
            size=lambda d: d["size"].astype(int),
        )
        children.append(html.H4("Портреты сегментов"))
        children.append(dash_table.DataTable(
            data=ports_view.to_dict("records"),
            columns=[{"name": c, "id": c} for c in
                     ["portrait", "size", "avg_price", "avg_duration", "top_format"]],
            style_cell={"fontSize": "13px", "padding": "6px"},
        ))

    children.append(html.H4("Гипотеза 1: наш прайс значимо ниже медианы рынка"))
    children.append(html.Ul([
        html.Li(f"n курсов: {h1['n']}"),
        html.Li(f"медиана рынка: {int(h1['median']):,} ₽".replace(",", " ")
                if h1.get("median") else "недостаточно данных"),
        html.Li(f"95% CI: {int(h1['ci_low']):,}..{int(h1['ci_high']):,} ₽".replace(",", " ")
                if h1.get("ci_low") else ""),
        html.Li(f"наш прайс: {int(h1['target']):,} ₽".replace(",", " ")),
        html.Li(f"Вывод: {h1['verdict']}"),
    ]))

    if corr is not None:
        children.append(html.H4("Гипотеза 2: цена и длительность положительно скоррелированы"))
        children.append(html.Ul([
            html.Li(f"Spearman ρ = {corr['rho']:.3f}"),
            html.Li(f"p-value = {corr['p_value']:.4f}, n = {corr['n']}"),
            html.Li(f"Вывод: {corr['verdict']}"),
        ]))
    return children


if __name__ == "__main__":
    make_app().run(debug=False, port=8050)
