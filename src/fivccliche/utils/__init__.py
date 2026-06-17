class UnsetType:
    """Sentinel type for omitted optional update fields."""

    def __repr__(self) -> str:
        return "UNSET"


UNSET = UnsetType()
