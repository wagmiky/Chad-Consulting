from __future__ import annotations

import numpy as np
import pandas as pd


class PriceAnalyzer:
    def __init__(self, frame: pd.DataFrame):
        self.df = frame.copy()
        self.df["price_rub"] = pd.to_numeric(self.df.get("price_rub"), errors="coerce")
        self.df["duration_months"] = pd.to_numeric(
            self.df.get("duration_months"), errors="coerce"
        )
        self.df["price_per_month"] = pd.to_numeric(
            self.df.get("price_per_month"), errors="coerce"
        )

    def summary(self) -> pd.DataFrame:
        """Базовая сводка по ценам в разрезе формата."""
        priced = self.df.dropna(subset=["price_rub"])
        if priced.empty:
            return pd.DataFrame()

        return (
            priced.groupby("format")["price_rub"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .round(0)
            .rename(
                columns={
                    "count": "n_courses",
                    "mean": "avg_price",
                    "median": "median_price",
                    "std": "price_std",
                    "min": "min_price",
                    "max": "max_price",
                }
            )
        )

    def price_segments(self, q=(0.25, 0.5, 0.75)) -> dict:
        priced = self.df.dropna(subset=["price_rub"])
        if priced.empty:
            return {}
        return {f"q{int(p * 100)}": float(np.quantile(priced["price_rub"], p)) for p in q}

    def under_budget(self, budget: float) -> pd.DataFrame:
        """Конкуренты, попадающие в наш ценовой коридор для студентов."""
        priced = self.df.dropna(subset=["price_rub"])
        return priced[priced["price_rub"] <= budget].sort_values("price_rub")

    def value_for_money(self) -> pd.DataFrame:
        """Сортировка по цене за месяц обучения."""
        df = self.df.dropna(subset=["price_per_month"])
        return df.sort_values("price_per_month")