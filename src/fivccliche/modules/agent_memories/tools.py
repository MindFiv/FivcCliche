from fivccliche.utils.deps import get_memory_provider_async


class MemoryRetain:
    """Store a piece of memory for the current user.

    Args:
        content: Memory text to retain.
    """

    def __init__(self, **context):
        self.user_uuid = context.get("user_uuid")

    async def __call__(self, content: str) -> str:
        if not self.user_uuid:
            raise ValueError("No user_uuid specified")

        mem_provider = await get_memory_provider_async()
        if mem_provider is None:
            raise ValueError("No memory provider specified")

        mem = mem_provider.get_memory(space_id=self.user_uuid)
        result = await mem.retain_async(content)
        return result.model_dump_json(exclude={"raw"})


class MemoryRecall:
    """Recall memories by semantic similarity for the current user.

    Args:
        query: Text to search memories against.
    """

    def __init__(self, **context):
        self.user_uuid = context.get("user_uuid")

    async def __call__(self, query: str) -> str:
        if not self.user_uuid:
            raise ValueError("No user_uuid specified")

        mem_provider = await get_memory_provider_async()
        if mem_provider is None:
            raise ValueError("No memory provider specified")

        mem = mem_provider.get_memory(space_id=self.user_uuid)
        result = await mem.recall_async(query)
        return result.model_dump_json(exclude={"raw"})


class MemoryList:
    """List memories for the current user.

    Args:
        skip: Number of memories to skip.
        limit: Maximum number of memories to return.
    """

    def __init__(self, **context):
        self.user_uuid = context.get("user_uuid")

    async def __call__(self, skip: int = 0, limit: int = 20) -> str:
        if not self.user_uuid:
            raise ValueError("No user_uuid specified")

        mem_provider = await get_memory_provider_async()
        if mem_provider is None:
            raise ValueError("No memory provider specified")

        mem = mem_provider.get_memory(space_id=self.user_uuid)
        result = await mem.list_async(skip=skip, limit=limit)
        return result.model_dump_json(exclude={"raw"})
