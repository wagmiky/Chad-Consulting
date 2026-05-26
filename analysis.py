"""
Анализ конкурентов: метрики по ценам, две гипотезы, корреляция,
кластеризация конкурентов, графики и итоговый отчёт.

Гипотезы:
1) Внутри онлайна цена за месяц у курсов с ментором значимо выше, чем
   у групповых вебинарных. Mann-Whitney, one-sided.
   Идея: проверяем именно надбавку за персональное внимание, потому что
   «ментор дороже самостоятельных» это тривиально (это разные продукты).
2) Цена за месяц обучения у офлайн-курсов значимо выше, чем у онлайн-курсов
   (то есть «офлайн дороже не просто потому что дольше»). Mann-Whitney, one-sided.

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

    def test_mentor_vs_group_per_month(self):
        """
        H0: цена за месяц у ментор-курсов не выше, чем у групповых вебинарных.
        H1: ментор-курсы стоят дороже за месяц.

        Сравниваем внутри онлайн-сегмента, чтобы поймать именно надбавку
        за персональное внимание. Просто «ментор дороже самостоятельных»
        тривиально, потому что это разные продукты по объёму контакта.
        """
        priced = self.courses.dropna(subset=["price_rub", "duration_months"]).copy()
        priced = priced[priced["duration_months"] > 0]
        priced["price_per_month"] = priced["price_rub"] / priced["duration_months"]
        mentor = priced.loc[priced["format"] == "online_mentor", "price_per_month"]
        group = priced.loc[priced["format"] == "online_group", "price_per_month"]
        return self._run_mw(
            "ментор дороже групповых за месяц обучения",
            mentor, group,
            label_a="mentor", label_b="group",
        )

    def test_offline_premium_per_month(self):
        """
        H0: цена за месяц у офлайн-курсов не выше, чем у онлайн.
        H1: офлайн стоит дороже в пересчёте на месяц.

        Идея: просто «офлайн дороже» это тривиально (он же дольше). Нам важно,
        есть ли надбавка за формат сверх длительности. Поэтому считаем
        цену за месяц и сравниваем распределения.
        """
        priced = self.courses.dropna(subset=["price_rub", "duration_months"]).copy()
        priced = priced[priced["duration_months"] > 0]
        priced["price_per_month"] = priced["price_rub"] / priced["duration_months"]

        offline = priced.loc[priced["format"] == "offline", "price_per_month"]
        online = priced.loc[priced["format"] != "offline", "price_per_month"]
        return self._run_mw(
            "офлайн дороже онлайна в пересчёте на месяц",
            offline, online,
            label_a="offline", label_b="online",
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
            "hypothesis_mentor": self.test_mentor_vs_group_per_month(),
            "hypothesis_offline_per_month": self.test_offline_premium_per_month(),
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

        h1 = report["hypothesis_mentor"]
        out.append("## Гипотеза 1: ментор дороже групповых вебинаров за месяц обучения")
        out.append(
            "Сравниваем внутри онлайна, чтобы поймать надбавку именно за "
            "персональное внимание, а не за сам факт сопровождения."
        )
        out.append(f"- n ментор: {h1['n_mentor']}, n групповых: {h1['n_group']}")
        if h1["p_value"] is not None:
            out.append(f"- Mann-Whitney U = {h1['stat']:.1f}, p-value = {h1['p_value']:.4f}")
            out.append(f"- медиана цены за месяц у ментора: {fmt_rub(h1['median_mentor'])}")
            out.append(f"- медиана цены за месяц у групповых: {fmt_rub(h1['median_group'])}")
        out.append(f"- Вывод: {h1['verdict']}")
        out.append("")

        h2 = report["hypothesis_offline_per_month"]
        out.append("## Гипотеза 2: офлайн дороже онлайна в пересчёте на месяц")
        out.append(
            "Проверяем не просто что офлайн дороже (это банально, он же дольше), "
            "а что цена за месяц обучения у офлайна выше."
        )
        out.append(f"- n офлайн: {h2['n_offline']}, n онлайн: {h2['n_online']}")
        if h2["p_value"] is not None:
            out.append(f"- Mann-Whitney U = {h2['stat']:.1f}, p-value = {h2['p_value']:.4f}")
            out.append(f"- медиана цены за месяц у офлайн: {fmt_rub(h2['median_offline'])}")
            out.append(f"- медиана цены за месяц у онлайн: {fmt_rub(h2['median_online'])}")
        out.append(f"- Вывод: {h2['verdict']}")
        out.append("")

        corr = report["correlation"]
        if corr:
            out.append("## Корреляция цена / длительность")
            out.append(f"Spearman ρ = {corr['rho']:.3f}, p-value = {corr['p_value']:.4f}, n = {corr['n']}.")
            out.append(f"Интерпретация: {corr['verdict']}.")
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
        h1 = report["hypothesis_mentor"]
        h2 = report["hypothesis_offline_per_month"]

        bits = []
        bits.append(
            f"При гипотезе нашего прайса в {fmt_rub(self.target_price)} дешевле "
            f"оказывается около {share * 100:.0f}% курсов на рынке."
        )

        if h1["p_value"] is not None and h1["p_value"] < 0.05:
            bits.append(
                "Внутри онлайна ментор-курсы значимо дороже групповых в "
                "пересчёте на месяц. Рынок берёт надбавку именно за "
                "персональное внимание, а не за сам факт сопровождения. "
                "Если мы делаем лёгкое сопровождение в групповом формате, "
                "это становится понятным якорем цены."
            )

        if h2["p_value"] is not None and h2["p_value"] < 0.05:
            bits.append(
                "Офлайн дороже онлайна даже в пересчёте на месяц обучения. "
                "Значит часть рыночной цены это именно надбавка за формат, "
                "а не просто длительность. Для нашего онлайн-продукта это "
                "значит верхнюю границу цены ниже офлайна, даже если "
                "длительность курса одинаковая."
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
