"""HTTP route registration for user-scoped config CRUD.

HTTP list/get returns every matching row, including inactive tools/skills.
Playground repositories filter ``is_active`` themselves so agents never pick up
disabled configs. Question configs have no playground repository.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, create_model
from pydantic_strict_partial import create_partial_model
from sqlalchemy.ext.asyncio import AsyncSession

from fivccliche.utils.asserts import assert_user_owns_resource
from fivccliche.utils.deps import IUser, get_authenticated_user_async, get_db_session_async
from fivccliche.utils.schemas import PaginatedResponse

BeforeUpdate = Callable[[Any, Any, IUser], None]
BeforeDelete = Callable[[Any, IUser], None]


@dataclass
class RouteConfig:
    """HTTP CRUD shape for one user-scoped config type."""

    slug: str
    noun: str
    schema: type[BaseModel]
    create_fn: Callable[..., Awaitable[Any]] | None = None
    get_fn: Callable[..., Awaitable[Any]] | None = None
    list_fn: Callable[..., Awaitable[list]] | None = None
    count_fn: Callable[..., Awaitable[int]] | None = None
    update_fn: Callable[..., Awaitable[Any]] | None = None
    delete_fn: Callable[..., Awaitable[None]] | None = None
    list_query: type[BaseModel] | None = None
    before_update: BeforeUpdate | None = None
    before_delete: BeforeDelete | None = None


def register_routes(router: APIRouter, spec: RouteConfig) -> None:
    """Register configured create/list/get/update/delete endpoints."""
    schema = spec.schema
    not_found = f"{spec.noun} not found"

    create_fn = spec.create_fn
    if create_fn is not None:

        @router.post(
            "/",
            summary=f"Create a new {spec.noun.lower()} for the authenticated user.",
            response_model=schema,
            status_code=status.HTTP_201_CREATED,
            operation_id=f"create_{spec.slug}",
        )
        async def create_config_async(
            config_create: schema,  # type: ignore[valid-type]
            user: IUser = Depends(get_authenticated_user_async),
            session: AsyncSession = Depends(get_db_session_async),
        ):
            config = await create_fn(
                session,
                None if user.is_superuser else user.uuid,
                config_create,
                updated_user_uuid=user.uuid,
            )
            return config.to_schema()

    list_fn = spec.list_fn
    count_fn = spec.count_fn
    if list_fn is not None and count_fn is not None:
        list_query = spec.list_query
        if list_query is not None:
            # FastAPI only flattens a Pydantic Query model when it is the sole
            # query object; skip/limit must live on the same model.
            list_params = create_model(
                f"{list_query.__name__}WithPaging",
                __base__=list_query,
                skip=(int, Field(0, ge=0)),
                limit=(int, Field(100, ge=1, le=1000)),
            )

            @router.get(
                "/",
                summary=f"List all {spec.noun.lower()}s for the authenticated user.",
                response_model=PaginatedResponse[schema],  # type: ignore[valid-type]
                operation_id=f"list_{spec.slug}s",
            )
            async def list_filtered_configs_async(
                query: Annotated[list_params, Query()],  # type: ignore[valid-type]
                user: IUser = Depends(get_authenticated_user_async),
                session: AsyncSession = Depends(get_db_session_async),
            ) -> PaginatedResponse:
                filters = query.model_dump()  # type: ignore[attr-defined]
                skip = filters.pop("skip")
                limit = filters.pop("limit")
                configs = await list_fn(session, user.uuid, skip=skip, limit=limit, **filters)
                total = await count_fn(session, user.uuid, **filters)
                return PaginatedResponse(
                    total=total,
                    results=[config.to_schema() for config in configs],
                )

        else:

            @router.get(
                "/",
                summary=f"List all {spec.noun.lower()}s for the authenticated user.",
                response_model=PaginatedResponse[schema],  # type: ignore[valid-type]
                operation_id=f"list_{spec.slug}s",
            )
            async def list_configs_async(
                skip: int = Query(0, ge=0),
                limit: int = Query(100, ge=1, le=1000),
                user: IUser = Depends(get_authenticated_user_async),
                session: AsyncSession = Depends(get_db_session_async),
            ) -> PaginatedResponse:
                configs = await list_fn(session, user.uuid, skip=skip, limit=limit)
                total = await count_fn(session, user.uuid)
                return PaginatedResponse(
                    total=total,
                    results=[config.to_schema() for config in configs],
                )

    get_fn = spec.get_fn
    if get_fn is not None:

        @router.get(
            "/{config_uuid}/",
            summary=f"Get a {spec.noun.lower()} by ID for the authenticated user.",
            response_model=schema,
            operation_id=f"get_{spec.slug}",
        )
        async def get_config_async(
            config_uuid: str,
            user: IUser = Depends(get_authenticated_user_async),
            session: AsyncSession = Depends(get_db_session_async),
        ):
            config = await get_fn(session, user.uuid, config_uuid=config_uuid)
            if not config:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=not_found)
            return config.to_schema()

    update_fn = spec.update_fn
    if get_fn is not None and update_fn is not None:

        @router.patch(
            "/{config_uuid}/",
            summary=f"Update a {spec.noun.lower()} by ID for the authenticated user.",
            response_model=schema,
            operation_id=f"update_{spec.slug}",
        )
        async def update_config_async(
            config_uuid: str,
            config_update: create_partial_model(schema),  # type: ignore[valid-type]
            user: IUser = Depends(get_authenticated_user_async),
            session: AsyncSession = Depends(get_db_session_async),
        ):
            config = await get_fn(session, user.uuid, config_uuid=config_uuid)
            if not config:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=not_found)
            assert_user_owns_resource(
                user,
                config.user_uuid,
                global_detail="Cannot update global configs",
                other_detail="Cannot update configs belonging to other users",
            )
            if spec.before_update:
                spec.before_update(config, config_update, user)
            config = await update_fn(session, config, config_update, updated_user_uuid=user.uuid)
            return config.to_schema()

    delete_fn = spec.delete_fn
    if get_fn is not None and delete_fn is not None:

        @router.delete(
            "/{config_uuid}/",
            summary=f"Delete a {spec.noun.lower()} by ID for the authenticated user.",
            status_code=status.HTTP_204_NO_CONTENT,
            operation_id=f"delete_{spec.slug}",
        )
        async def delete_config_async(
            config_uuid: str,
            user: IUser = Depends(get_authenticated_user_async),
            session: AsyncSession = Depends(get_db_session_async),
        ) -> None:
            config = await get_fn(session, user.uuid, config_uuid=config_uuid)
            if not config:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=not_found)
            assert_user_owns_resource(
                user,
                config.user_uuid,
                global_detail="Cannot delete global configs",
                other_detail="Cannot delete configs belonging to other users",
            )
            if spec.before_delete:
                spec.before_delete(config, user)
            await delete_fn(session, config)
