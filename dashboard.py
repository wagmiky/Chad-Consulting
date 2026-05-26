"""
Интерактивный дашборд на Dash. Три вкладки: обзор по школам,
карта позиционирования и кластеры с гипотезами.

Запуск: python dashboard.py
"""

import plotly.express as px
from dash import Dash, dcc, html, dash_table, Input, Output

from collector import CourseScraper, YouTubeFetcher
from analysis import CompetitorAnalysis


def _prepare():
    courses = CourseScraper().load()
    youtube = YouTubeFetcher(channels=[]).load()  # из кеша
    analysis = CompetitorAnalysis(courses, youtube)
    return courses, youtube, analysis


def make_app():
    courses, youtube, analysis = _prepare()

    fmt_options = [{"label": "Все форматы", "value": "all"}]
    for f in sorted(courses["format"].dropna().unique()):
        fmt_options.append({"label": f, "value": f})

    h1 = analysis.test_edtech_vs_traditional_per_month()
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
                            title="Цена и длительность",
                            template="simple_white",
                            labels={"duration_months": "Длительность, мес",
                                    "price_rub": "Цена, руб"},
                        )
                    ),
                    dcc.Graph(
                        figure=px.bar(
                            youtube.sort_values("subscribers", ascending=False),
                            x="subscribers",
                            y="channel",
                            orientation="h",
                            title="YouTube-конкуренты по числу подписчиков",
                            template="simple_white",
                        ).update_layout(yaxis={"categoryorder": "total ascending"})
                    ),
                ]),

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

    children.append(html.H4("Гипотеза 1: edtech дороже традиционных школ за месяц"))
    children.append(html.Ul([
        html.Li(f"n: edtech={h1['n_edtech']}, традиционные={h1['n_trad']}"),
        html.Li(f"Mann-Whitney p-value: {h1['p_value']:.4f}"
                if h1["p_value"] is not None else "недостаточно данных"),
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
