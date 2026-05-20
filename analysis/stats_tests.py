from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    sample_sizes: dict
    conclusion: str

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


class HypothesisRunner:
    """Набор гипотез, которые нам интересны для позиционирования."""

    def __init__(self, frame: pd.DataFrame):
        self.df = frame.copy()
        self.df["price_rub"] = pd.to_numeric(self.df.get("price_rub"), errors="coerce")
        self.df["duration_months"] = pd.to_numeric(
            self.df.get("duration_months"), errors="coerce"
        )

    # Гипотеза 1: цена индивидуальных курсов с ментором выше, чем у самостоятельных.
    def mentor_vs_self_price(self) -> TestResult:
        mentor = self._prices_where(format="online_mentor")
        self_paced = self._prices_where(format="online_self")
        return self._mann_whitney(
            "Курсы с ментором дороже самостоятельных",
            mentor,
            self_paced,
        )

    # Гипотеза 2: длительные курсы (от 3 месяцев) системно дороже коротких.
    def long_vs_short_price(self) -> TestResult:
        priced = self.df.dropna(subset=["price_rub", "duration_months"])
        long_ = priced.loc[priced["duration_months"] >= 3, "price_rub"]
        short = priced.loc[priced["duration_months"] < 3, "price_rub"]
        return self._mann_whitney(
            "Длинные курсы дороже коротких",
            long_,
            short,
        )

    # Гипотеза 3: цены отличаются между всеми форматами одновременно.
    def price_across_formats(self) -> TestResult:
        priced = self.df.dropna(subset=["price_rub", "format"])
        groups = [g["price_rub"].values for _, g in priced.groupby("format")]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            return TestResult(
                "Различия цен между форматами",
                statistic=float("nan"),
                p_value=float("nan"),
                sample_sizes={},
                conclusion="Слишком мало данных для теста",
            )
        stat, p = stats.kruskal(*groups)
        sizes = {f"group_{i + 1}": len(g) for i, g in enumerate(groups)}
        return TestResult(
            name="Различия цен между форматами",
            statistic=float(stat),
            p_value=float(p),
            sample_sizes=sizes,
            conclusion=_describe_kruskal(p),
        )

    def bootstrap_median(self, values: Iterable[float], n: int = 2000) -> tuple[float, float, float]:
        """95% доверительный интервал для медианы методом бутстрепа."""
        arr = np.array([v for v in values if not np.isnan(v)])
        if len(arr) < 5:
            return float("nan"), float("nan"), float("nan")
        rng = np.random.default_rng(42)
        medians = [np.median(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
        return (
            float(np.median(arr)),
            float(np.quantile(medians, 0.025)),
            float(np.quantile(medians, 0.975)),
        )

    def run_all(self) -> pd.DataFrame:
        results = [
            self.mentor_vs_self_price(),
            self.long_vs_short_price(),
            self.price_across_formats(),
        ]
        return pd.DataFrame(
            [
                {
                    "hypothesis": r.name,
                    "statistic": round(r.statistic, 3) if not np.isnan(r.statistic) else None,
                    "p_value": round(r.p_value, 4) if not np.isnan(r.p_value) else None,
                    "significant_at_5pct": r.is_significant(),
                    "samples": r.sample_sizes,
                    "conclusion": r.conclusion,
                }
                for r in results
            ]
        )

    def _prices_where(self, **filters) -> pd.Series:
        df = self.df.dropna(subset=["price_rub"])
        for column, value in filters.items():
            df = df[df[column] == value]
        return df["price_rub"]

    def _mann_whitney(self, name: str, a: pd.Series, b: pd.Series) -> TestResult:
        a = a.dropna()
        b = b.dropna()
        if len(a) < 3 or len(b) < 3:
            return TestResult(
                name=name,
                statistic=float("nan"),
                p_value=float("nan"),
                sample_sizes={"a": len(a), "b": len(b)},
                conclusion="Выборка слишком мала, тест неинформативен",
            )
        stat, p = stats.mannwhitneyu(a, b, alternative="greater")
        return TestResult(
            name=name,
            statistic=float(stat),
            p_value=float(p),
            sample_sizes={"a": len(a), "b": len(b)},
            conclusion=_describe_mw(p),
        )


def _describe_mw(p: float) -> str:
    if p < 0.01:
        return "Различия выраженные, влияние формата на цену видно отчётливо"
    if p < 0.05:
        return "Различия значимы, но эффект умеренный"
    return "На текущей выборке отличий не видно"


def _describe_kruskal(p: float) -> str:
    if p < 0.05:
        return "Хотя бы одна группа выпадает из общего паттерна цен"
    return "Цены по форматам сопоставимы, отличий нет"