"""Hindsight-backed implementation of IUserMemoryProvider.

Reads connection settings from the ``hindsight`` config session in
``.env.json`` (keys ``BASE_URL`` / ``API_KEY`` / ``TIMEOUT``) and wraps the
``hindsight_client.Hindsight`` SDK, mapping its native responses onto the
implementation-agnostic ``MemoryContent`` / ``MemoryRetainResult`` /
``MemoryRecallResult`` models.
"""

from __future__ import annotations

import logging
from typing import Any

from fivcglue import IComponentSite, query_component
from fivcglue.interfaces import configs
from hindsight_client import Hindsight

from fivccliche.services.interfaces.agent_memories import (
    IUserMemory,
    IUserMemoryProvider,
    MemoryContent,
    MemoryRecallResult,
    MemoryRetainResult,
)

logger = logging.getLogger(__name__)

_DEFAULT_BANK_ID = "default"
_DEFAULT_BASE_URL = "http://localhost:8888"
_DEFAULT_TIMEOUT = 300.0


class UserMemoryHindsightImpl(IUserMemory):
    """IUserMemory backed by a single Hindsight memory bank."""

    def __init__(self, hindsight: Hindsight, bank_id: str) -> None:
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

    @staticmethod
    def _map_item(r: Any) -> MemoryContent:
        item_type = getattr(r, "type", None)
        return MemoryContent(
            id=getattr(r, "id", None),
            content=getattr(r, "text", "") or "",
            score=getattr(r, "score", None),
            categories=[item_type] if item_type else None,
            metadata=getattr(r, "metadata", None),
            created_at=getattr(r, "timestamp", None),
        )


class UserMemoryProviderImpl(IUserMemoryProvider):
    """Provider that hands out UserMemoryHindsightImpl bound to a Hindsight client.

    The Hindsight client is created lazily on first ``get_memory`` call so that
    component loading does not perform any I/O, and so that a missing config
    session only surfaces when memory is actually used.
    """

    def __init__(self, component_site: IComponentSite, **_kwargs: Any) -> None:
        self._component_site = component_site
        self._hindsight: Hindsight | None = None

    def _get_hindsight(self) -> Hindsight:
        if self._hindsight is None:
            self._hindsight = self._build_hindsight()
        return self._hindsight

    def _build_hindsight(self) -> Hindsight:
        session = self._read_config_session()
        base_url = (session.get_value("BASE_URL") if session else None) or _DEFAULT_BASE_URL
        api_key = session.get_value("API_KEY") if session else None
        timeout_raw = session.get_value("TIMEOUT") if session else None
        timeout = float(timeout_raw) if timeout_raw else _DEFAULT_TIMEOUT
        logger.info("Initializing Hindsight memory client (base_url=%s)", base_url)
        return Hindsight(base_url=base_url, api_key=api_key, timeout=timeout)

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
