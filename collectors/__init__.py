from .base import BaseCollector
from .static_courses import TutortopCollector
from .youtube_dynamic import YouTubeCollector
from .vk_api import VKGroupsCollector

__all__ = [
    "BaseCollector",
    "TutortopCollector",
    "YouTubeCollector",
    "VKGroupsCollector",
]