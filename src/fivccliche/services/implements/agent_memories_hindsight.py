"""Hindsight-backed implementation of IUserMemoryProvider.

Reads connection settings from the ``hindsight`` config session in
``.env.json`` (keys ``BASE_URL`` / ``API_KEY`` / ``TIMEOUT``) and wraps the
``hindsight_client.Hindsight`` SDK, mapping its native responses onto the
implementation-agnostic ``MemoryContent`` / ``MemoryRetainResult`` /
``MemoryRecallResult`` models.

Requires the optional ``hindsight-client`` package. Install it separately
before mounting this provider.
"""

from __future__ import annotations

import logging
from typing import Any

from fivcglue import IComponentSite, query_component
from fivcglue.interfaces import configs

from fivccliche.services.interfaces.agent_memories import (
    IUserMemory,
    IUserMemoryProvider,
    MemoryContent,
    MemoryListResult,
    MemoryRecallResult,
    MemoryRetainResult,
)
from fivccliche.utils.parsers import to_float

logger = logging.getLogger(__name__)

_DEFAULT_BANK_ID = "default"
_DEFAULT_BASE_URL = "http://localhost:8888"
_DEFAULT_TIMEOUT = 300.0


def _import_hindsight() -> type:
    try:
        from hindsight_client import Hindsight
    except ImportError as exc:
        raise ImportError(
            "hindsight-client is required to use UserMemoryProviderImpl. "
            "Install it with: pip install hindsight-client"
        ) from exc
    return Hindsight


class UserMemoryHindsightImpl(IUserMemory):
    """IUserMemory backed by a single Hindsight memory bank."""

    def __init__(self, hindsight: Any, bank_id: str) -> None:
        self._hindsight = hindsight
        self._bank_id = bank_id

    async def retain_async(self, content: str) -> MemoryRetainResult:
        resp = await self._hindsight.aretain(bank_id=self._bank_id, content=content)
        success = bool(getattr(resp, "success", True))
        return MemoryRetainResult(success=success, count=1, raw=resp)

    async def recall_async(self, query: str) -> MemoryRecallResult:
        resp = await self._hindsight.arecall(bank_id=self._bank_id, query=query)
        raw_items = getattr(resp, "results", None) or []
        items = [self._map_item(r) for r in raw_items]
        return MemoryRecallResult(items=items, raw=resp)

    async def list_async(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        **kwargs: Any,
    ) -> MemoryListResult:
        resp = await self._hindsight.memory.list_memories(
            bank_id=self._bank_id,
            type=kwargs.get("type"),
            q=kwargs.get("search_query") or kwargs.get("q"),
            limit=limit,
            offset=skip,
        )
        raw_items = getattr(resp, "items", None) or []
        items = [self._map_item(r) for r in raw_items]
        total = int(getattr(resp, "total", 0) or 0)
        return MemoryListResult(items=items, total=total, raw=resp)

    @staticmethod
    def _attr(r: Any, name: str, default: Any = None) -> Any:
        if isinstance(r, dict):
            return r.get(name, default)
        return getattr(r, name, default)

    @classmethod
    def _map_item(cls, r: Any) -> MemoryContent:
        item_type = cls._attr(r, "type")
        content = cls._attr(r, "text") or cls._attr(r, "content") or ""
        created_at = cls._attr(r, "timestamp")
        if created_at is None:
            created_at = cls._attr(r, "created_at")
        return MemoryContent(
            id=cls._attr(r, "id"),
            content=content,
            score=cls._attr(r, "score"),
            categories=[item_type] if item_type else None,
            metadata=cls._attr(r, "metadata"),
            created_at=created_at,
        )


class UserMemoryProviderImpl(IUserMemoryProvider):
    """Provider that hands out UserMemoryHindsightImpl bound to a Hindsight client.

    Imports ``hindsight-client`` when the component is constructed so a missing
    optional dependency fails fast at mount time. The Hindsight client instance
    is still created lazily on first ``get_memory`` call so component loading
    does not perform any I/O.
    """

    def __init__(self, component_site: IComponentSite, **_kwargs: Any) -> None:
        self._component_site = component_site
        self._hindsight_cls = _import_hindsight()
        self._hindsight: Any | None = None

    def _get_hindsight(self) -> Any:
        if self._hindsight is None:
            self._hindsight = self._build_hindsight()
        return self._hindsight

    def _build_hindsight(self) -> Any:
        session = self._read_config_session()
        base_url = (session.get_value("BASE_URL") if session else None) or _DEFAULT_BASE_URL
        api_key = session.get_value("API_KEY") if session else None
        timeout = to_float(session.get_value("TIMEOUT") if session else None, _DEFAULT_TIMEOUT)
        logger.info("Initializing Hindsight memory client (base_url=%s)", base_url)
        return self._hindsight_cls(base_url=base_url, api_key=api_key, timeout=timeout)

    def _read_config_session(self) -> configs.IConfigSession | None:
        config = query_component(self._component_site, configs.IConfig)
        if config is None:
            logger.warning("IConfig component not registered; using Hindsight defaults")
            return None
        session = config.get_session("hindsight")
        if session is None:
            logger.warning("Config session 'hindsight' not found; using Hindsight defaults")
        return session

    def get_memory(
        self,
        space_id: str | None = None,
        **_kwargs: Any,
    ) -> IUserMemory:
        bank_id = space_id or _DEFAULT_BANK_ID
        return UserMemoryHindsightImpl(self._get_hindsight(), bank_id)
