"""Интерактивный дашборд на Dash.

Запуск: `python -m visualization.dashboard`. Дашборд берёт данные из
финального CSV (`competitors_clean.csv`) и собирает три вкладки: обзор,
карта позиционирования и проверка гипотез.
"""

from __future__ import annotations

import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output

from analysis.stats_tests import HypothesisRunner
from config import CLEAN_COMPETITORS_CSV
from visualization.plots import PlotBuilder


def _load() -> pd.DataFrame:
    if not CLEAN_COMPETITORS_CSV.exists():
        raise FileNotFoundError(
            "Сначала прогоните `python main.py --full`, чтобы появился чистый набор данных."
        )
    return pd.read_csv(CLEAN_COMPETITORS_CSV)


def build_app() -> Dash:
    df = _load()
    plots = PlotBuilder(df)
    hyp = HypothesisRunner(df).run_all()

    app = Dash(__name__, title="Конкуренты: внешний вид")

    fmt_options = [{"label": "Все форматы", "value": "all"}]
    for fmt in sorted(df["format"].dropna().unique()):
        fmt_options.append({"label": fmt, "value": fmt})

    app.layout = html.Div(
        style={"fontFamily": "Inter, system-ui, sans-serif", "padding": "24px"},
        children=[
            html.H1("Анализ конкурентов: онлайн-курсы по внешнему виду"),
            html.P(
                "Данные собраны из каталога tutortop, метаданных YouTube-каналов и "
                "публичных групп ВКонтакте. Выберите формат, чтобы отфильтровать таблицу."
            ),
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Обзор рынка",
                        children=[
                            dcc.Graph(figure=plots.audience_bar() or _empty_fig()),
                            dcc.Dropdown(
                                id="format-filter",
                                options=fmt_options,
                                value="all",
                                clearable=False,
                                style={"width": "300px", "marginTop": "16px"},
                            ),
                            dash_table.DataTable(
                                id="table",
                                columns=[
                                    {"name": c, "id": c}
                                    for c in [
                                        "name",
                                        "source",
                                        "format",
                                        "price_rub",
                                        "duration_months",
                                        "audience_size",
                                    ]
                                    if c in df.columns
                                ],
                                page_size=15,
                                style_table={"overflowX": "auto"},
                                style_cell={"fontSize": "13px", "padding": "6px"},
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Карта позиционирования",
                        children=[dcc.Graph(figure=plots.positioning_map() or _empty_fig())],
                    ),
                    dcc.Tab(
                        label="Гипотезы",
                        children=[
                            html.H3("Что показала статистика"),
                            dash_table.DataTable(
                                data=hyp.to_dict("records"),
                                columns=[{"name": c, "id": c} for c in hyp.columns],
                                style_cell={"fontSize": "13px", "padding": "6px"},
                                style_data_conditional=[
                                    {
                                        "if": {"filter_query": "{significant_at_5pct} = True"},
                                        "backgroundColor": "#E8F6EE",
                                    }
                                ],
                            ),
                        ],
                    ),
                ]
            ),
        ],
    )

    @app.callback(Output("table", "data"), Input("format-filter", "value"))
    def filter_table(value):
        if value == "all":
            view = df
        else:
            view = df[df["format"] == value]
        return view.to_dict("records")

    return app


def _empty_fig():
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        annotations=[
            dict(
                text="Данных пока нет, попробуйте перезапустить пайплайн",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
        ],
        template="simple_white",
    )
    return fig


if __name__ == "__main__":
    build_app().run(debug=False, port=8050)