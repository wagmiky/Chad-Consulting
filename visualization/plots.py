from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


PALETTE = {
    "online_self": "#5B8DEF",
    "online_group": "#F3B53F",
    "online_mentor": "#E36A6A",
    "offline": "#7D8597",
    "personal_brand": "#9C6ADE",
    "community": "#4FB286",
}


class PlotBuilder:
    def __init__(self, frame: pd.DataFrame):
        self.df = frame.copy()
        self.df["price_rub"] = pd.to_numeric(self.df.get("price_rub"), errors="coerce")
        self.df["duration_months"] = pd.to_numeric(
            self.df.get("duration_months"), errors="coerce"
        )

    def price_distribution(self, save_to: Path | None = None):
        priced = self.df.dropna(subset=["price_rub", "format"])
        if priced.empty:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        formats = priced["format"].unique()
        data = [priced.loc[priced["format"] == f, "price_rub"] for f in formats]
        bp = ax.boxplot(data, labels=formats, patch_artist=True)
        for patch, fmt in zip(bp["boxes"], formats):
            patch.set_facecolor(PALETTE.get(fmt, "#888"))
            patch.set_alpha(0.8)

        ax.set_title("Распределение цен по форматам обучения")
        ax.set_ylabel("Цена, руб")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()

        if save_to:
            fig.savefig(save_to, dpi=150)
        return fig

    def positioning_map(self):
        """Карта позиционирования: цена vs длительность, размер по охвату."""
        df = self.df.dropna(subset=["price_rub", "duration_months"]).copy()
        if df.empty:
            return None
        df["audience_size"] = pd.to_numeric(df.get("audience_size"), errors="coerce")
        df["audience_size"] = df["audience_size"].fillna(df["audience_size"].median() or 1000)

        fig = px.scatter(
            df,
            x="duration_months",
            y="price_rub",
            size="audience_size",
            color="format",
            hover_name="name",
            color_discrete_map=PALETTE,
            size_max=40,
            title="Карта позиционирования конкурентов",
            labels={
                "duration_months": "Длительность, мес.",
                "price_rub": "Цена, руб.",
                "format": "Формат",
            },
        )
        fig.update_layout(template="simple_white")
        return fig

    def audience_bar(self):
        df = self.df.dropna(subset=["audience_size"]).copy()
        if df.empty:
            return None
        top = df.sort_values("audience_size", ascending=False).head(15)
        fig = px.bar(
            top,
            x="audience_size",
            y="name",
            orientation="h",
            color="source",
            title="Топ-15 конкурентов по размеру аудитории",
            labels={"audience_size": "Подписчики / участники", "name": "Конкурент"},
        )
        fig.update_layout(template="simple_white", yaxis={"categoryorder": "total ascending"})
        return fig