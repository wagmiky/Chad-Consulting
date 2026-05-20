from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import logging
import time

import pandas as pd


log = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Скелет сборщика. Наследники реализуют `_fetch`."""

    source_name: str = "base"

    def __init__(self, cache_path: Path, refresh: bool = False, throttle: float = 1.0):
        self.cache_path = Path(cache_path)
        self.refresh = refresh
        self.throttle = throttle  # пауза между запросами, чтобы не злить сайт

    def collect(self) -> pd.DataFrame:
        """Главная точка входа. Возвращает таблицу с сырыми данными.

        Если есть кеш и нас не просили обновляться, читаем его. Иначе тянем из
        источника, сохраняем в CSV и отдаём дальше.
        """
        if self.cache_path.exists() and not self.refresh:
            log.info("%s: читаю кеш %s", self.source_name, self.cache_path.name)
            return pd.read_csv(self.cache_path)

        log.info("%s: иду в сеть", self.source_name)
        rows = list(self._fetch())
        df = pd.DataFrame(rows)
        df["source"] = self.source_name
        df.to_csv(self.cache_path, index=False, encoding="utf-8-sig")
        return df

    def _sleep(self) -> None:
        if self.throttle:
            time.sleep(self.throttle)

    @abstractmethod
    def _fetch(self):
        """Должен быть генератором словарей с данными по конкурентам."""
        raise NotImplementedError