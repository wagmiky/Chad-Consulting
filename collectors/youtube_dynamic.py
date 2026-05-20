from __future__ import annotations

from typing import Iterator

from yt_dlp import YoutubeDL

from config import YOUTUBE_CHANNELS
from .base import BaseCollector


YDL_OPTS = {
    "quiet": True,
    "skip_download": True,
    "extract_flat": "in_playlist",
    "playlistend": 25,  # последние 25 роликов на канал, нам хватит для оценки
    "no_warnings": True,
}


class YouTubeCollector(BaseCollector):
    source_name = "youtube"

    def __init__(self, cache_path, refresh: bool = False, channels=None):
        super().__init__(cache_path, refresh=refresh, throttle=0.5)
        self.channels = channels or YOUTUBE_CHANNELS

    def _fetch(self) -> Iterator[dict]:
        with YoutubeDL(YDL_OPTS) as ydl:
            for channel in self.channels:
                try:
                    info = ydl.extract_info(channel + "/videos", download=False)
                except Exception as exc:
                    # Не падаем из-за одного канала: пишем пометку и идём дальше.
                    yield {"channel": channel, "error": str(exc)}
                    continue

                entries = info.get("entries", []) or []
                views = [e.get("view_count") or 0 for e in entries]
                likes = [e.get("like_count") or 0 for e in entries]

                avg_views = sum(views) / len(views) if views else 0
                avg_likes = sum(likes) / len(likes) if likes else 0
                engagement = (avg_likes / avg_views) if avg_views else 0

                yield {
                    "channel": info.get("channel") or info.get("uploader"),
                    "channel_url": channel,
                    "subscribers": info.get("channel_follower_count"),
                    "videos_sampled": len(entries),
                    "avg_views": int(avg_views),
                    "avg_likes": int(avg_likes),
                    "engagement_rate": round(engagement, 4),
                }

                self._sleep()