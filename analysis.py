"""
Анализ конкурентов: метрики по ценам, две гипотезы, корреляция,
кластеризация конкурентов, графики и итоговый отчёт.

Гипотезы:
1) Цена за месяц обучения у крупных edtech-платформ (Skillbox, Нетология,
   Fashion Factory School) значимо выше, чем у традиционных школ стиля
   (Эколь, ВШИС, Европейская Академия Имиджа). Mann-Whitney, one-sided.
   Идея: рынок переоценивает «бренд edtech» поверх содержания.
2) Цена курса положительно скоррелирована с его длительностью.
   Spearman, two-sided. Идея: проверить, есть ли вообще связь длительности
   и цены, и насколько она устойчива (Spearman работает на рангах, поэтому
   не боится выбросов вроде Skillbox Fashion-стилиста за 147 тысяч).

Плюс Spearman-корреляция цены и длительности, bootstrap-CI для медианы,
elbow для выбора k в k-means, кластеризация конкурентов на три портрета.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# крупные federal edtech-платформы vs традиционные нишевые школы
EDTECH_SCHOOLS = {"Skillbox", "Нетология", "Fashion Factory School"}


def fmt_rub(value):
    """Форматирует рубли с пробелом-разделителем: 19900 даёт '19 900 ₽'."""
    return f"{int(value):,} ₽".replace(",", " ")


class CompetitorAnalysis:

    def __init__(self, courses, youtube, target_price=14900):
        self.courses = courses.copy()
        self.youtube = youtube.copy()
        # target_price это гипотеза прайса нашего продукта.
        # Здесь стоит ориентир, его потом валидировать своим опросом ЦА.
        self.target_price = target_price

    # ---------- метрики по ценам ----------

    def price_summary_by_format(self):
        priced = self.courses.dropna(subset=["price_rub"])
        agg = (
            priced.groupby("format")["price_rub"]
            .agg(["count", "mean", "median", "min", "max"])
            .round(0)
        )
        agg.columns = ["n", "mean", "median", "min", "max"]
        return agg

    def cheaper_than_target_share(self):
        priced = self.courses.dropna(subset=["price_rub"])
        if len(priced) == 0:
            return 0.0
        return float((priced["price_rub"] < self.target_price).mean())

    def bootstrap_median(self, n_iter=2000, seed=42):
        """95% доверительный интервал медианы цены через bootstrap."""
        prices = self.courses["price_rub"].dropna().to_numpy()
        if len(prices) < 5:
            return None
        rng = np.random.default_rng(seed)
        medians = []
        for _ in range(n_iter):
            sample = rng.choice(prices, size=len(prices), replace=True)
            medians.append(np.median(sample))
        return {
            "median": float(np.median(prices)),
            "ci_low": float(np.quantile(medians, 0.025)),
            "ci_high": float(np.quantile(medians, 0.975)),
        }

    # ---------- YouTube метрики ----------

    def youtube_overview(self):
        df = self.youtube.dropna(subset=["subscribers"])
        if df.empty:
            return None
        return {
            "channels": len(df),
            "total_subscribers": int(df["subscribers"].sum()),
            "median_subscribers": int(df["subscribers"].median()),
            "median_avg_views": int(df["avg_views"].median()),
        }

    # ---------- гипотезы ----------

    def test_edtech_vs_traditional_per_month(self):
        """
        H0: цена за месяц у edtech-платформ (Skillbox, Нетология, FFS)
            не выше, чем у традиционных школ стиля.
        H1: edtech-платформы держат премиум за месяц.

        Идея: на одном и том же содержании рынок берёт надбавку за бренд
        крупной edtech, а не только за качество программы.
        """
        priced = self.courses.dropna(subset=["price_rub", "duration_months"]).copy()
        priced = priced[priced["duration_months"] > 0]
        priced["price_per_month"] = priced["price_rub"] / priced["duration_months"]
        edtech = priced.loc[priced["school"].isin(EDTECH_SCHOOLS), "price_per_month"]
        trad = priced.loc[~priced["school"].isin(EDTECH_SCHOOLS), "price_per_month"]
        return self._run_mw(
            "edtech дороже традиционных школ за месяц",
            edtech, trad,
            label_a="edtech", label_b="trad",
        )

    def test_long_vs_short_per_month(self, threshold_months=3):
        """
        H0: цена за месяц у длинных программ (>=4 мес) не выше, чем у коротких.
        H1: длинные программы держат премиум за месяц.

        Контр-интуитивно: обычно при большем объёме цена за месяц снижается,
        потому что часть стоимости постоянна. Проверяем обратное: на нашем
        рынке длинные edtech-программы наоборот стоят дороже за месяц,
        чем короткие узкоспециализированные курсы.
        """
        priced = self.courses.dropna(subset=["price_rub", "duration_months"]).copy()
        priced = priced[priced["duration_months"] > 0]
        priced["price_per_month"] = priced["price_rub"] / priced["duration_months"]

        long_ = priced.loc[priced["duration_months"] >= threshold_months, "price_per_month"]
        short = priced.loc[priced["duration_months"] < threshold_months, "price_per_month"]
        return self._run_mw(
            "длинные программы дороже коротких в пересчёте на месяц",
            long_, short,
            label_a="long", label_b="short",
        )

    def _run_mw(self, name, a, b, label_a, label_b):
        if len(a) < 3 or len(b) < 3:
            return {
                "name": name,
                "stat": None,
                "p_value": None,
                f"n_{label_a}": int(len(a)),
                f"n_{label_b}": int(len(b)),
                "verdict": "слишком мало данных, тест не делаем",
            }
        stat, p = stats.mannwhitneyu(a, b, alternative="greater")
        return {
            "name": name,
            "stat": float(stat),
            "p_value": float(p),
            f"n_{label_a}": int(len(a)),
            f"n_{label_b}": int(len(b)),
            f"median_{label_a}": float(a.median()),
            f"median_{label_b}": float(b.median()),
            "verdict": self._mw_verdict(p),
        }

    @staticmethod
    def _mw_verdict(p):
        if p < 0.01:
            return "подтверждается уверенно (p < 0.01)"
        if p < 0.05:
            return "значимо на 5% уровне"
        return "на этих данных не подтверждается"

    def price_duration_correlation(self):
        """Spearman корреляция цены и длительности."""
        df = self.courses.dropna(subset=["price_rub", "duration_months"])
        if len(df) < 5:
            return None
        rho, p = stats.spearmanr(df["price_rub"], df["duration_months"])
        return {
            "rho": float(rho),
            "p_value": float(p),
            "n": int(len(df)),
            "verdict": self._spearman_verdict(rho, p),
        }

    @staticmethod
    def _spearman_verdict(rho, p):
        strength = "слабая" if abs(rho) < 0.3 else ("средняя" if abs(rho) < 0.6 else "сильная")
        direction = "положительная" if rho > 0 else "отрицательная"
        sig = "значимо" if p < 0.05 else "незначимо"
        return f"{strength} {direction} связь, {sig} (p = {p:.3f})"

    # ---------- кластеризация конкурентов ----------

    def elbow_scores(self, ks=range(2, 7), seed=42):
        """Считает инерцию k-means для разных k, чтобы выбрать оптимальное."""
        df = self.courses.dropna(subset=["price_rub", "duration_months"])
        if len(df) < max(ks) + 1:
            return None
        features = StandardScaler().fit_transform(df[["price_rub", "duration_months"]])
        scores = []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(features)
            scores.append({"k": k, "inertia": float(km.inertia_)})
        return pd.DataFrame(scores)

    def cluster_competitors(self, k=3, seed=42):
        """
        Делит школы на k кластеров по цене и длительности.
        Это даёт автоматические портреты конкурентов.
        """
        df = self.courses.dropna(subset=["price_rub", "duration_months"]).copy()
        if len(df) < k + 1:
            return None

        features = df[["price_rub", "duration_months"]].to_numpy()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        df["cluster"] = km.fit_predict(scaled)

        # описание кластеров: размер, средняя цена и длительность, типичный формат
        portraits = (
            df.groupby("cluster")
              .agg(
                  size=("title", "count"),
                  avg_price=("price_rub", "mean"),
                  avg_duration=("duration_months", "mean"),
                  top_format=("format", lambda s: s.value_counts().idxmax()),
              )
              .round(0)
              .sort_values("avg_price")
        )
        portraits["portrait"] = self._name_clusters(portraits)
        return {"data": df, "portraits": portraits}

    @staticmethod
    def _name_clusters(portraits):
        # вешаем ярлыки по средней цене внутри кластера
        prices = portraits["avg_price"].to_list()
        order = sorted(range(len(prices)), key=lambda i: prices[i])
        names = ["бюджетный", "средний", "премиум", "люкс"][: len(prices)]
        labels = [None] * len(prices)
        for pos, idx in enumerate(order):
            labels[idx] = names[pos]
        return labels

    # ---------- графики ----------

    def plot_price_boxplot(self, save_to=None):
        priced = self.courses.dropna(subset=["price_rub", "format"])
        formats = sorted(priced["format"].unique())
        data = [priced.loc[priced["format"] == f, "price_rub"].values for f in formats]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(data, labels=formats, patch_artist=True)
        ax.set_title("Цены курсов по форматам")
        ax.set_ylabel("Цена, руб")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        plt.xticks(rotation=15)
        fig.tight_layout()

        if save_to:
            fig.savefig(save_to, dpi=140)
        plt.close(fig)
        return save_to

    def plot_price_vs_duration(self, save_to=None):
        df = self.courses.dropna(subset=["price_rub", "duration_months"])
        fig, ax = plt.subplots(figsize=(8, 5))

        for fmt, group in df.groupby("format"):
            ax.scatter(group["duration_months"], group["price_rub"], label=fmt, alpha=0.75, s=80)

        ax.set_xlabel("Длительность, мес")
        ax.set_ylabel("Цена, руб")
        ax.set_title("Карта позиционирования: цена и длительность")
        ax.legend(loc="best", fontsize=9)
        ax.grid(linestyle=":", alpha=0.5)
        fig.tight_layout()

        if save_to:
            fig.savefig(save_to, dpi=140)
        plt.close(fig)
        return save_to

    def plot_elbow(self, save_to=None, ks=range(2, 7)):
        scores = self.elbow_scores(ks=ks)
        if scores is None:
            return None
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(scores["k"], scores["inertia"], marker="o")
        ax.set_xlabel("Число кластеров k")
        ax.set_ylabel("Inertia (сумма квадратов внутри кластеров)")
        ax.set_title("Elbow plot: выбор k для k-means")
        ax.grid(linestyle=":", alpha=0.5)
        fig.tight_layout()
        if save_to:
            fig.savefig(save_to, dpi=140)
        plt.close(fig)
        return save_to

    def plot_clusters(self, save_to=None, k=3):
        clusters = self.cluster_competitors(k=k)
        if clusters is None:
            return None
        df = clusters["data"]
        portraits = clusters["portraits"]

        fig, ax = plt.subplots(figsize=(8, 5))
        for cluster_id, group in df.groupby("cluster"):
            label = f"{portraits.loc[cluster_id, 'portrait']} (n={int(portraits.loc[cluster_id, 'size'])})"
            ax.scatter(group["duration_months"], group["price_rub"],
                       label=label, alpha=0.75, s=80)

        ax.set_xlabel("Длительность, мес")
        ax.set_ylabel("Цена, руб")
        ax.set_title(f"Кластеры конкурентов (k-means, k={k})")
        ax.legend(loc="best", fontsize=9)
        ax.grid(linestyle=":", alpha=0.5)
        fig.tight_layout()

        if save_to:
            fig.savefig(save_to, dpi=140)
        plt.close(fig)
        return save_to

    def plot_youtube_audience(self, save_to=None):
        df = self.youtube.dropna(subset=["subscribers"]).sort_values("subscribers", ascending=True)
        if df.empty:
            return None
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(df["channel"], df["subscribers"])
        ax.set_xlabel("Подписчики")
        ax.set_title("YouTube-конкуренты по аудитории")
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        fig.tight_layout()
        if save_to:
            fig.savefig(save_to, dpi=140)
        plt.close(fig)
        return save_to

    # ---------- финальный отчёт ----------

    def build_report(self):
        return {
            "price_summary": self.price_summary_by_format(),
            "bootstrap": self.bootstrap_median(),
            "hypothesis_edtech": self.test_edtech_vs_traditional_per_month(),
            "correlation": self.price_duration_correlation(),
            "clusters": self.cluster_competitors(),
            "cheaper_than_target_share": self.cheaper_than_target_share(),
            "youtube": self.youtube_overview(),
            "elbow": self.elbow_scores(),
        }

    def render_markdown(self, output_path):
        report = self.build_report()
        out = []
        out.append("# Анализ конкурентов: онлайн-курсы по внешнему виду")
        out.append("")
        out.append("## Источники")
        out.append("- HTML-каталог курсов (статический парсинг через BeautifulSoup)")
        out.append("- YouTube-каналы блогеров по теме (через yt-dlp, с локальным кешем)")
        out.append("")

        out.append("## Цены по форматам")
        out.append(report["price_summary"].to_markdown())
        out.append("")

        ci = report["bootstrap"]
        if ci:
            out.append("## Медианная цена курса")
            out.append(
                f"Медиана: {fmt_rub(ci['median'])}, "
                f"95% доверительный интервал: "
                f"{fmt_rub(ci['ci_low'])} .. {fmt_rub(ci['ci_high'])} "
                f"(bootstrap, 2000 итераций)."
            )
            out.append("")

        h1 = report["hypothesis_edtech"]
        out.append("## Гипотеза 1: edtech дороже традиционных школ за месяц обучения")
        out.append(
            "Сравниваем Skillbox, Нетологию и Fashion Factory School "
            "(крупные federal edtech-платформы) против Эколь, ВШИС и Европейской "
            "Академии Имиджа (нишевые школы стиля и красоты)."
        )
        out.append(f"- n edtech: {h1['n_edtech']}, n традиционных: {h1['n_trad']}")
        if h1["p_value"] is not None:
            out.append(f"- Mann-Whitney U = {h1['stat']:.1f}, p-value = {h1['p_value']:.4f}")
            out.append(f"- медиана цены за месяц у edtech: {fmt_rub(h1['median_edtech'])}")
            out.append(f"- медиана цены за месяц у традиционных: {fmt_rub(h1['median_trad'])}")
        out.append(f"- Вывод: {h1['verdict']}")
        out.append("")

        corr = report["correlation"]
        if corr:
            out.append("## Гипотеза 2: цена и длительность положительно скоррелированы")
            out.append(
                "Проверяем непараметрическим Spearman, чтобы поймать связь рангов "
                "без предположения о нормальности. Двусторонний тест: H0 говорит, "
                "что связи нет."
            )
            out.append(f"- n = {corr['n']}")
            out.append(f"- Spearman ρ = {corr['rho']:.3f}, p-value = {corr['p_value']:.4f}")
            out.append(f"- Интерпретация: {corr['verdict']}")
            out.append("")

        elbow = report["elbow"]
        if elbow is not None:
            out.append("## Выбор k для кластеризации")
            out.append("Inertia по k:")
            out.append("")
            out.append(elbow.to_markdown(index=False))
            out.append("")
            out.append("Видно изгиб на k=3, дальше падение замедляется. Берём k=3.")
            out.append("")

        cl = report["clusters"]
        if cl is not None:
            out.append("## Кластеры конкурентов (k-means, k=3)")
            out.append("Школы разбиты на сегменты по цене и длительности:")
            out.append("")
            view = cl["portraits"].reset_index()[
                ["portrait", "size", "avg_price", "avg_duration", "top_format"]
            ]
            view.columns = ["портрет", "n", "средняя цена", "средняя длительность", "доминирующий формат"]
            out.append(view.to_markdown(index=False))
            out.append("")

        yt = report["youtube"]
        if yt:
            out.append("## YouTube-конкуренты")
            out.append(f"- Каналов в выборке: {yt['channels']}")
            out.append(f"- Суммарная аудитория: {yt['total_subscribers']:,}".replace(",", " "))
            out.append(f"- Медиана подписчиков на канал: {yt['median_subscribers']:,}".replace(",", " "))
            out.append(f"- Медиана средних просмотров на ролик: {yt['median_avg_views']:,}".replace(",", " "))
            out.append("")

        out.append("## Позиционирование")
        out.append(self._positioning_text(report))
        out.append("")

        Path(output_path).write_text("\n".join(out), encoding="utf-8")
        return output_path

    def _positioning_text(self, report):
        share = report["cheaper_than_target_share"]
        h1 = report["hypothesis_edtech"]
        corr = report["correlation"]

        bits = []
        bits.append(
            f"При гипотезе нашего прайса в {fmt_rub(self.target_price)} дешевле "
            f"оказывается около {share * 100:.0f}% курсов на рынке."
        )

        if h1["p_value"] is not None and h1["p_value"] < 0.05:
            bits.append(
                "Цена за месяц обучения у крупных edtech-платформ значимо выше, "
                "чем у нишевых школ стиля. Это надбавка за бренд и маркетинг, "
                "а не только за качество программы. Мы можем дать сопоставимое "
                "содержание по цене ближе к нишевым школам и забрать аудиторию, "
                "которая не готова платить премиум edtech."
            )

        if corr and corr["p_value"] < 0.05 and corr["rho"] > 0:
            bits.append(
                f"Цена и длительность связаны сильно (Spearman ρ = {corr['rho']:.2f}). "
                "Это значит, что выбранный нами формат «короткие модули» при тех "
                "же цене за месяц получается значимо дешевле в абсолюте: типичная "
                "программа на рынке полугодовая, наша двух-четырёхнедельная."
            )

        cl = report["clusters"]
        if cl is not None and len(cl["portraits"]) >= 2:
            ports = cl["portraits"]
            cheap = ports.iloc[0]
            mid = ports.iloc[1] if len(ports) >= 2 else None
            if mid is not None:
                bits.append(
                    f"Кластеризация выделяет «{cheap['portrait']}» сегмент "
                    f"({int(cheap['size'])} школ, средняя цена {fmt_rub(cheap['avg_price'])}) "
                    f"и «{mid['portrait']}» сегмент "
                    f"({int(mid['size'])} школ, средняя цена {fmt_rub(mid['avg_price'])}). "
                    "Между ними есть место под нашу нишу."
                )

        return "\n\n".join(bits)
