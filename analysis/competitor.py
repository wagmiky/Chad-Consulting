from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Iterator

import pandas as pd


@dataclass
class Competitor:
    name: str
    source: str
    price_rub: float | None = None
    duration_months: float | None = None
    duration_hours: float | None = None
    format: str | None = None
    audience_size: int | None = None
    engagement_rate: float | None = None
    url: str | None = None
    notes: list[str] = field(default_factory=list)

    def price_per_month(self) -> float | None:
        if self.price_rub and self.duration_months:
            return round(self.price_rub / self.duration_months, 2)
        return None

    def is_priced(self) -> bool:
        return self.price_rub is not None and self.price_rub > 0

    def as_row(self) -> dict:
        d = asdict(self)
        d["price_per_month"] = self.price_per_month()
        d["notes"] = "; ".join(self.notes) if self.notes else ""
        return d


class CompetitorRegistry:
    """Хранит конкурентов и умеет фильтровать по разным признакам."""

    def __init__(self, competitors: Iterable[Competitor] | None = None):
        self._items: list[Competitor] = list(competitors or [])

    def add(self, competitor: Competitor) -> None:
        self._items.append(competitor)

    def extend(self, items: Iterable[Competitor]) -> None:
        self._items.extend(items)

    def __iter__(self) -> Iterator[Competitor]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.as_row() for c in self._items])

    def by_format(self, fmt: str) -> "CompetitorRegistry":
        return CompetitorRegistry(c for c in self._items if c.format == fmt)

    def priced_only(self) -> "CompetitorRegistry":
        return CompetitorRegistry(c for c in self._items if c.is_priced())

    @classmethod
    def from_frames(
        cls,
        courses: pd.DataFrame,
        youtube: pd.DataFrame,
        vk: pd.DataFrame,
    ) -> "CompetitorRegistry":
        """Склеивает три источника в единый список конкурентов.

        Курсы это основа: один курс одно предложение. YouTube и VK добавляют
        информацию об аудитории. Сопоставляем по нормализованному имени школы
        или автора.
        """
        registry = cls()

        for _, row in courses.iterrows():
            name = (row.get("school") or row.get("title") or "").strip()
            if not name:
                continue
            registry.add(
                Competitor(
                    name=name,
                    source="tutortop",
                    price_rub=_to_float(row.get("price_rub")),
                    duration_months=_to_float(row.get("duration_months")),
                    duration_hours=_to_float(row.get("duration_hours")),
                    format=row.get("format"),
                    url=row.get("url"),
                )
            )

        # Авторские каналы как отдельные конкуренты: они тоже продают свои
        # курсы, просто через личный бренд, а не через школу.
        for _, row in youtube.iterrows():
            name = row.get("channel")
            if not name or pd.isna(name):
                continue
            registry.add(
                Competitor(
                    name=str(name),
                    source="youtube",
                    audience_size=_to_int(row.get("subscribers")),
                    engagement_rate=_to_float(row.get("engagement_rate")),
                    url=row.get("channel_url"),
                    format="personal_brand",
                )
            )

        for _, row in vk.iterrows():
            name = row.get("name")
            if not name or pd.isna(name):
                continue
            registry.add(
                Competitor(
                    name=str(name),
                    source="vk",
                    audience_size=_to_int(row.get("members")),
                    url=f"https://vk.com/{row.get('screen_name')}" if row.get("screen_name") else None,
                    format="community",
                    notes=[row.get("activity")] if row.get("activity") else [],
                )
            )

        return registry


def _to_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value) -> int | None:
    f = _to_float(value)
    return int(f) if f is not None else None