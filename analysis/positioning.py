from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pricing import PriceAnalyzer
from .stats_tests import HypothesisRunner


class PositioningReport:
    def __init__(self, frame: pd.DataFrame, our_price_target: float = 9900):
        self.df = frame
        self.target_price = our_price_target
        self.pricing = PriceAnalyzer(frame)
        self.stats = HypothesisRunner(frame)

    def build(self) -> dict:
        summary = self.pricing.summary()
        segments = self.pricing.price_segments()
        hypotheses = self.stats.run_all()

        priced = self.df.dropna(subset=["price_rub"])
        median, lo, hi = self.stats.bootstrap_median(priced["price_rub"])

        budget_competitors = self.pricing.under_budget(self.target_price)
        cheaper_share = (
            len(budget_competitors) / len(priced) if len(priced) else 0
        )

        return {
            "summary_by_format": summary,
            "price_quantiles": segments,
            "market_median_price": median,
            "market_median_ci": (lo, hi),
            "competitors_in_our_budget": budget_competitors,
            "share_cheaper_than_us": round(cheaper_share, 3),
            "hypotheses": hypotheses,
        }

    def render_markdown(self, output_path: Path) -> Path:
        report = self.build()
        lines = ["# Отчёт по конкурентам", ""]

        lines.append("## Базовая сводка по форматам")
        if not report["summary_by_format"].empty:
            lines.append(report["summary_by_format"].to_markdown())
        else:
            lines.append("Цен в выборке пока недостаточно для разбивки.")
        lines.append("")

        lines.append("## Ценовые сегменты рынка")
        for q, value in report["price_quantiles"].items():
            lines.append(f"- {q}: {int(value):,} ₽".replace(",", " "))
        lines.append("")

        med = report["market_median_price"]
        lo, hi = report["market_median_ci"]
        if med == med:  # not NaN
            lines.append(
                f"Медианная цена курса на рынке примерно {int(med):,} ₽ "
                f"(95% доверительный интервал по бутстрепу: {int(lo):,} ₽ — {int(hi):,} ₽).".replace(
                    ",", " "
                )
            )
        lines.append("")

        lines.append("## Сколько конкурентов дешевле нашего таргета")
        lines.append(
            f"При цене {self.target_price:,} ₽ дешевле нас оказывается "
            f"{report['share_cheaper_than_us'] * 100:.1f}% конкурентов.".replace(",", " ")
        )
        lines.append("")

        lines.append("## Проверка гипотез")
        lines.append(report["hypotheses"].to_markdown(index=False))
        lines.append("")

        lines.append("## Вывод по позиционированию")
        lines.append(self._positioning_text(report))

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def _positioning_text(self, report: dict) -> str:
        # Подбираем формулировку под фактические данные. Если рынок дорогой,
        # выгодно играть на доступности; если дешёвый, акцент на качестве.
        med = report["market_median_price"]
        share = report["share_cheaper_than_us"]
        if med != med:
            return (
                "Данных по ценам пока мало. Стоит расширить парсинг или брать "
                "альтернативный каталог, чтобы получить более устойчивые цифры."
            )

        if share < 0.3:
            angle = (
                "Наша цена ниже медианы рынка, поэтому уместно строить позиционирование "
                "вокруг доступной цены и быстрого результата для студентов."
            )
        elif share > 0.7:
            angle = (
                "Большинство конкурентов дешевле, поэтому нужна сильная содержательная "
                "дифференциация: персональное сопровождение и фокус на луксмаксинге, "
                "которого нет у массовых школ стиля."
            )
        else:
            angle = (
                "Мы попадаем в середину рынка по цене, поэтому отстраивать нас стоит "
                "через формат: короткие модули плюс личный куратор там, где у конкурентов "
                "большие потоки без обратной связи."
            )

        return angle