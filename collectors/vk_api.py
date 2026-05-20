from __future__ import annotations

from typing import Iterator

import requests

from config import SEARCH_QUERIES, VK_API_VERSION, VK_TOKEN
from .base import BaseCollector


VK_API = "https://api.vk.com/method"


class VKGroupsCollector(BaseCollector):
    source_name = "vk"

    def __init__(self, cache_path, refresh: bool = False, per_query: int = 20):
        super().__init__(cache_path, refresh=refresh, throttle=0.4)
        self.per_query = per_query

    def _call(self, method: str, params: dict) -> dict:
        params = {**params, "access_token": VK_TOKEN, "v": VK_API_VERSION}
        r = requests.get(f"{VK_API}/{method}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def _fetch(self) -> Iterator[dict]:
        if not VK_TOKEN:
            # Возвращаем явный маркер, чтобы было видно: источник пропущен.
            yield {"group_id": None, "name": None, "note": "VK_TOKEN не задан"}
            return

        seen = set()
        for query in SEARCH_QUERIES:
            data = self._call(
                "groups.search",
                {"q": query, "count": self.per_query, "type": "group"},
            )
            items = data.get("response", {}).get("items", [])
            ids = [str(item["id"]) for item in items if item.get("id") not in seen]
            if not ids:
                continue
            seen.update(ids)

            details = self._call(
                "groups.getById",
                {
                    "group_ids": ",".join(ids),
                    "fields": "members_count,description,activity,verified",
                },
            )

            for group in details.get("response", {}).get("groups", []):
                yield {
                    "query": query,
                    "group_id": group.get("id"),
                    "name": group.get("name"),
                    "screen_name": group.get("screen_name"),
                    "members": group.get("members_count"),
                    "activity": group.get("activity"),
                    "verified": bool(group.get("verified")),
                    "description": (group.get("description") or "")[:300],
                }

            self._sleep()