import asyncio
import base64
import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    responses,
    status,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import ValidationError
from pydantic_strict_partial import create_partial_model
from sqlalchemy.ext.asyncio import AsyncSession

from fivcplayground.tools import create_tool_retriever_async

from fivccliche.services.interfaces.agent_configs import IUserConfigProvider
from fivccliche.services.interfaces.agent_speeches import (
    SpeechAudioInput,
    SpeechRecognizeOptions,
    SpeechRequestError,
    SpeechSynthesisOptions,
)
from fivccliche.utils.deps import (
    IUser,
    get_authenticated_user_async,
    get_authenticator_async,
    get_config_provider_async,
    get_db_session_async,
    get_speech_provider_async,
)
from fivccliche.utils import deps
from fivccliche.utils.schemas import PaginatedResponse

from . import models, schemas, utils
from .filters import QuestionFilterSet, UserScopedEditableFilterSet, UserScopedReadableFilterSet


def _reject_frozen_agent_update(config, config_update, _user) -> None:
    if not config.is_frozen:
        return
    fields_set: set[str] = getattr(config_update, "model_fields_set", set())
    if fields_set - {"is_frozen"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Agent config is frozen and cannot be edited",
        )


def _reject_frozen_agent_delete(config, _user) -> None:
    if config.is_frozen:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Agent config is frozen and cannot be deleted",
        )


# ============================================================================
# Embedding Config Endpoints
# ============================================================================

router_embeddings = APIRouter(prefix="/configs/embeddings", tags=["embedding_configs"])


@router_embeddings.post(
    "/",
    summary="Create a new embedding config for the authenticated user.",
    response_model=schemas.UserEmbeddingSchema,
    status_code=status.HTTP_201_CREATED,
    operation_id="create_embedding_config",
)
async def create_embedding_config_async(
    config_create: schemas.UserEmbeddingSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await utils.create_embedding_config_async(
        session,
        None if user.is_superuser else user.uuid,
        config_create,
        updated_user_uuid=user.uuid,
    )
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_embeddings.get(
    "/",
    summary="List all embedding configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserEmbeddingSchema],
    operation_id="list_embedding_configs",
)
async def list_embedding_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse:
    filters = UserScopedReadableFilterSet(
        models.UserEmbedding.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    configs = await utils.list_user_scoped_async(
        session, models.UserEmbedding, filters=filters, skip=skip, limit=limit
    )
    total = await utils.count_user_scoped_async(session, models.UserEmbedding, filters=filters)
    return PaginatedResponse(
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_embeddings.get(
    "/{config_uuid}/",
    summary="Get a embedding config by ID for the authenticated user.",
    response_model=schemas.UserEmbeddingSchema,
    operation_id="get_embedding_config",
)
async def get_embedding_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedReadableFilterSet(
        models.UserEmbedding.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserEmbedding, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Embedding config not found"
        )
    return config.to_schema()


@router_embeddings.patch(
    "/{config_uuid}/",
    summary="Update a embedding config by ID for the authenticated user.",
    response_model=schemas.UserEmbeddingSchema,
    operation_id="update_embedding_config",
)
async def update_embedding_config_async(
    config_uuid: str,
    config_update: create_partial_model(schemas.UserEmbeddingSchema),  # type: ignore[valid-type]
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedEditableFilterSet(
        models.UserEmbedding.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserEmbedding, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Embedding config not found"
        )
    config = await utils.update_embedding_config_async(
        session, config, config_update, updated_user_uuid=user.uuid
    )
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_embeddings.delete(
    "/{config_uuid}/",
    summary="Delete a embedding config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="delete_embedding_config",
)
async def delete_embedding_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    filters = UserScopedEditableFilterSet(
        models.UserEmbedding.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserEmbedding, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Embedding config not found"
        )
    await session.delete(config)
    await session.commit()


# ============================================================================
# LLM Config Endpoints
# ============================================================================

router_models = APIRouter(prefix="/configs/models", tags=["model_configs"])


@router_models.post(
    "/",
    summary="Create a new llm config for the authenticated user.",
    response_model=schemas.UserLLMSchema,
    status_code=status.HTTP_201_CREATED,
    operation_id="create_llm_config",
)
async def create_llm_config_async(
    config_create: schemas.UserLLMSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await utils.create_llm_config_async(
        session,
        None if user.is_superuser else user.uuid,
        config_create,
        updated_user_uuid=user.uuid,
    )
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_models.get(
    "/",
    summary="List all llm configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserLLMSchema],
    operation_id="list_llm_configs",
)
async def list_llm_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse:
    filters = UserScopedReadableFilterSet(
        models.UserLLM.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    configs = await utils.list_user_scoped_async(
        session, models.UserLLM, filters=filters, skip=skip, limit=limit
    )
    total = await utils.count_user_scoped_async(session, models.UserLLM, filters=filters)
    return PaginatedResponse(
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_models.get(
    "/{config_uuid}/",
    summary="Get a llm config by ID for the authenticated user.",
    response_model=schemas.UserLLMSchema,
    operation_id="get_llm_config",
)
async def get_llm_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedReadableFilterSet(
        models.UserLLM.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserLLM, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LLM config not found")
    return config.to_schema()


@router_models.patch(
    "/{config_uuid}/",
    summary="Update a llm config by ID for the authenticated user.",
    response_model=schemas.UserLLMSchema,
    operation_id="update_llm_config",
)
async def update_llm_config_async(
    config_uuid: str,
    config_update: create_partial_model(schemas.UserLLMSchema),  # type: ignore[valid-type]
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedEditableFilterSet(
        models.UserLLM.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserLLM, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LLM config not found")
    config = await utils.update_llm_config_async(
        session, config, config_update, updated_user_uuid=user.uuid
    )
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_models.delete(
    "/{config_uuid}/",
    summary="Delete a llm config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="delete_llm_config",
)
async def delete_llm_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    filters = UserScopedEditableFilterSet(
        models.UserLLM.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserLLM, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="LLM config not found")
    await session.delete(config)
    await session.commit()


# ============================================================================
# ASR Config Endpoints
# ============================================================================

router_asrs = APIRouter(prefix="/configs/asrs", tags=["asr_configs"])


@router_asrs.post(
    "/",
    summary="Create a new asr config for the authenticated user.",
    response_model=schemas.UserASRSchema,
    status_code=status.HTTP_201_CREATED,
    operation_id="create_asr_config",
)
async def create_asr_config_async(
    config_create: schemas.UserASRSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await utils.create_asr_config_async(
        session,
        None if user.is_superuser else user.uuid,
        config_create,
        updated_user_uuid=user.uuid,
    )
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_asrs.get(
    "/",
    summary="List all asr configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserASRSchema],
    operation_id="list_asr_configs",
)
async def list_asr_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse:
    filters = UserScopedReadableFilterSet(
        models.UserASR.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    configs = await utils.list_user_scoped_async(
        session, models.UserASR, filters=filters, skip=skip, limit=limit
    )
    total = await utils.count_user_scoped_async(session, models.UserASR, filters=filters)
    return PaginatedResponse(
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_asrs.post(
    "/{config_uuid}/probe/",
    summary="Probe asr config for the authenticated user.",
    status_code=status.HTTP_200_OK,
)
async def probe_asr_config_async(
    config_uuid: str,
    probe: schemas.UserASRProbeRequest,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> responses.StreamingResponse:
    filters = UserScopedReadableFilterSet(
        models.UserASR.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserASR, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ASR config not found")
    model_type = config.model_type
    api_key = config.api_key
    model = config.model
    base_url = config.base_url
    await session.close()

    speech_provider = await get_speech_provider_async(model_type)
    if speech_provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speech provider is not mounted",
        )

    options = SpeechRecognizeOptions(
        language=probe.language,
        hotwords=probe.hotwords,
        context=probe.context,
        format=probe.format,
    )

    async def stream():
        try:
            async with await speech_provider.get_recognizer(
                options,
                api_key=api_key,
                model=model,
                base_url=base_url,
            ) as recognizer:
                async for event in recognizer.stream_async(
                    SpeechAudioInput(url=probe.url, data_b64=probe.data_b64, format=probe.format),
                ):
                    if event.type == "error":
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "event": "error",
                                    "info": {
                                        "message": event.message or "Speech recognition failed"
                                    },
                                }
                            )
                            + "\n\n"
                        )
                        return
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "event": event.type,
                                "info": {"text": event.text, "language": event.language},
                            }
                        )
                        + "\n\n"
                    )
        except SpeechRequestError as exc:
            yield (
                "data: "
                + json.dumps(
                    {
                        "event": "error",
                        "info": {"message": str(exc) or "Speech recognition failed"},
                    }
                )
                + "\n\n"
            )

    return responses.StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router_asrs.websocket(
    "/{config_uuid}/probe/",
)
async def probe_realtime_asr_config_async(
    config_uuid: str,
    websocket: WebSocket,
) -> None:
    from fivccliche.utils.chats import ChatChannel, ChatChannelError

    authenticator = await get_authenticator_async()
    async with ChatChannel(websocket, authenticator) as channel:
        user = await channel.authenticate_async()
        async with deps.get_db_session_context_async() as session:
            filters = UserScopedReadableFilterSet(
                models.UserASR.user_uuid, user.uuid, is_superuser=user.is_superuser
            )
            config = await utils.get_user_scoped_async(
                session, models.UserASR, filters=filters, config_uuid=config_uuid
            )
            model_type = config.model_type if config else None
            api_key = config.api_key if config else None
            asr_model = config.model if config else None
            base_url = config.base_url if config else None

        if config is None:
            await channel.fail_async(
                code="asr_not_found",
                message="ASR config not found",
                close_code=4404,
            )
            return
        if model_type != "dashscope_realtime":
            await channel.fail_async(
                code="invalid_model_type",
                message="Realtime ASR is required",
                close_code=1003,
            )
            return

        start_frame = await channel.receive_json_async()
        if start_frame is None:
            return
        if start_frame.get("type") != "start":
            await channel.fail_async(
                code="invalid_start",
                message="A start frame is required",
                close_code=1003,
            )
        if start_frame.get("format") != "pcm" or start_frame.get("sample_rate") != 16000:
            await channel.fail_async(
                code="invalid_start",
                message="PCM audio at 16000 Hz is required",
                close_code=1003,
            )
            return

        speech_provider = await get_speech_provider_async(model_type)
        if speech_provider is None:
            await channel.fail_async(
                code="provider_unmounted",
                message="Speech provider is not mounted",
                close_code=1013,
            )
            return

        options = SpeechRecognizeOptions(
            language=start_frame.get("language"),
            hotwords=start_frame.get("hotwords"),
            context=start_frame.get("context"),
            format="pcm",
            sample_rate=16000,
        )
        try:
            recognizer = await speech_provider.get_recognizer(
                options,
                api_key=api_key,
                model=asr_model,
                base_url=base_url,
            )
        except SpeechRequestError as exc:
            await channel.fail_async(
                code="recognition_failed",
                message=str(exc) or "Speech recognition failed",
                close_code=1011,
            )
            return

        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        output_queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def _read_audio() -> None:
            while True:
                frame = await channel.receive_async()
                if frame is None:
                    break
                if isinstance(frame, bytes):
                    await audio_queue.put(frame)
                    continue
                if isinstance(frame, str):
                    try:
                        frame = json.loads(frame)
                    except json.JSONDecodeError:
                        frame = None
                if not isinstance(frame, dict) or frame.get("type") != "audio_commit":
                    await output_queue.put(
                        {
                            "event": "error",
                            "info": {
                                "code": "invalid_frame",
                                "message": "Binary PCM or audio_commit is required",
                            },
                        }
                    )
                    break
                break
            await audio_queue.put(None)

        async def _recognize() -> None:
            async def _audio() -> AsyncIterator[bytes]:
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        return
                    yield chunk

            try:
                async with recognizer:
                    await channel.send_json_async({"event": "ready", "info": {}})
                    final_parts: list[str] = []
                    async for event in recognizer.stream_async(_audio()):
                        payload = {"text": event.text, "language": event.language}
                        if event.type == "error":
                            await output_queue.put(
                                {
                                    "event": "error",
                                    "info": {
                                        "code": "recognition_failed",
                                        "message": event.message or "Speech recognition failed",
                                    },
                                }
                            )
                            return
                        await output_queue.put({"event": event.type, "info": payload})
                        if event.type == "final":
                            final_parts.append(event.text)
                    await output_queue.put(
                        {"event": "complete", "info": {"text": "".join(final_parts)}}
                    )
            except Exception as exc:
                message = (
                    str(exc) if isinstance(exc, SpeechRequestError) else "Speech recognition failed"
                )
                await output_queue.put(
                    {
                        "event": "error",
                        "info": {"code": "recognition_failed", "message": message},
                    }
                )
            finally:
                await output_queue.put(None)

        audio_task = asyncio.create_task(_read_audio())
        recognition_task = asyncio.create_task(_recognize())
        try:
            while True:
                item = await output_queue.get()
                if item is None:
                    return
                await channel.send_json_async(item)
                if item.get("event") == "error":
                    message = str(item.get("info", {}).get("message", ""))
                    raise ChatChannelError(code=1011, msg=message)
                if item.get("event") == "complete":
                    return
        except (RuntimeError, WebSocketDisconnect):
            return
        finally:
            audio_task.cancel()
            recognition_task.cancel()
            await asyncio.gather(audio_task, recognition_task, return_exceptions=True)


@router_asrs.get(
    "/{config_uuid}/",
    summary="Get a asr config by ID for the authenticated user.",
    response_model=schemas.UserASRSchema,
    operation_id="get_asr_config",
)
async def get_asr_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedReadableFilterSet(
        models.UserASR.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserASR, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ASR config not found")
    return config.to_schema()


@router_asrs.patch(
    "/{config_uuid}/",
    summary="Update a asr config by ID for the authenticated user.",
    response_model=schemas.UserASRSchema,
    operation_id="update_asr_config",
)
async def update_asr_config_async(
    config_uuid: str,
    config_update: create_partial_model(schemas.UserASRSchema),  # type: ignore[valid-type]
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedEditableFilterSet(
        models.UserASR.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserASR, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ASR config not found")
    config = await utils.update_asr_config_async(
        session, config, config_update, updated_user_uuid=user.uuid
    )
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_asrs.delete(
    "/{config_uuid}/",
    summary="Delete a asr config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="delete_asr_config",
)
async def delete_asr_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    filters = UserScopedEditableFilterSet(
        models.UserASR.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserASR, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ASR config not found")
    await session.delete(config)
    await session.commit()


# ============================================================================
# TTS Config Endpoints
# ============================================================================

router_tts = APIRouter(prefix="/configs/tts", tags=["tts_configs"])


@router_tts.post(
    "/",
    summary="Create a new tts config for the authenticated user.",
    response_model=schemas.UserTTSSchema,
    status_code=status.HTTP_201_CREATED,
    operation_id="create_tts_config",
)
async def create_tts_config_async(
    config_create: schemas.UserTTSSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await utils.create_tts_config_async(
        session,
        None if user.is_superuser else user.uuid,
        config_create,
        updated_user_uuid=user.uuid,
    )
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_tts.get(
    "/",
    summary="List all tts configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserTTSSchema],
    operation_id="list_tts_configs",
)
async def list_tts_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse:
    filters = UserScopedReadableFilterSet(
        models.UserTTS.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    configs = await utils.list_user_scoped_async(
        session, models.UserTTS, filters=filters, skip=skip, limit=limit
    )
    total = await utils.count_user_scoped_async(session, models.UserTTS, filters=filters)
    return PaginatedResponse(
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_tts.post(
    "/{config_uuid}/probe/",
    summary="Probe tts config for the authenticated user.",
    status_code=status.HTTP_200_OK,
)
async def probe_tts_config_async(
    config_uuid: str,
    probe: schemas.UserTTSProbeRequest,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> responses.StreamingResponse:
    filters = UserScopedReadableFilterSet(
        models.UserTTS.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserTTS, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="TTS config not found")
    api_key = config.api_key
    model = config.model
    base_url = config.base_url
    await session.close()

    speech_provider = await get_speech_provider_async(config.model_type)
    if speech_provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speech provider is not mounted",
        )
    options = SpeechSynthesisOptions(
        voice=probe.voice,
        format=probe.format,
        sample_rate=probe.sample_rate,
        volume=probe.volume,
        speech_rate=probe.speech_rate,
        pitch_rate=probe.pitch_rate,
    )

    async def stream():
        try:
            async with await speech_provider.get_synthesizer(
                options,
                api_key=api_key,
                model=model,
                base_url=base_url,
            ) as synthesizer:
                async for audio in synthesizer.stream_async(probe.text):
                    encoded = base64.b64encode(audio).decode("ascii")
                    yield "data: " + json.dumps(
                        {"event": "audio", "info": {"data_b64": encoded}}
                    ) + "\n\n"
            yield "data: " + json.dumps({"event": "complete", "info": {}}) + "\n\n"
        except SpeechRequestError as exc:
            yield (
                "data: " + json.dumps({"event": "error", "info": {"message": str(exc)}}) + "\n\n"
            )

    return responses.StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router_tts.websocket("/{config_uuid}/probe/")
async def probe_realtime_tts_config_async(
    config_uuid: str,
    websocket: WebSocket,
) -> None:
    """Stream one synthesis probe over the realtime provider WebSocket protocol."""
    from fivccliche.utils.chats import ChatChannel

    authenticator = await get_authenticator_async()
    async with ChatChannel(websocket, authenticator) as channel:
        user = await channel.authenticate_async()

        async with deps.get_db_session_context_async() as session:
            filters = UserScopedReadableFilterSet(
                models.UserTTS.user_uuid, user.uuid, is_superuser=user.is_superuser
            )
            config = await utils.get_user_scoped_async(
                session, models.UserTTS, filters=filters, config_uuid=config_uuid
            )
            model_type = config.model_type if config else None
            api_key = config.api_key if config else None
            tts_model = config.model if config else None
            base_url = config.base_url if config else None

        if config is None:
            await channel.fail_async(
                code="tts_not_found",
                message="TTS config not found",
                close_code=4404,
            )
            return
        if model_type != "dashscope_realtime":
            await channel.fail_async(
                code="invalid_model_type",
                message="Realtime TTS is required",
                close_code=1003,
            )
            return

        start_frame = await channel.receive_json_async()
        if start_frame is None:
            return
        if start_frame.get("type") != "start":
            await channel.fail_async(
                code="invalid_start",
                message="A start frame is required",
                close_code=1003,
            )
        try:
            probe = schemas.UserTTSProbeRequest.model_validate(start_frame)
        except ValidationError:
            await channel.fail_async(
                code="invalid_start",
                message="Invalid TTS probe parameters",
                close_code=1003,
            )
            return

        speech_provider = await get_speech_provider_async(model_type)
        if speech_provider is None:
            await channel.fail_async(
                code="provider_unmounted",
                message="Speech provider is not mounted",
                close_code=1013,
            )
            return

        options = SpeechSynthesisOptions(
            voice=probe.voice,
            format=probe.format,
            sample_rate=probe.sample_rate,
            volume=probe.volume,
            speech_rate=probe.speech_rate,
            pitch_rate=probe.pitch_rate,
        )
        try:
            synthesizer = await speech_provider.get_synthesizer(
                options,
                api_key=api_key,
                model=tts_model,
                base_url=base_url,
            )
            async with synthesizer:
                await channel.send_json_async({"event": "ready", "info": {}})
                async for audio in synthesizer.stream_async(probe.text):
                    encoded = base64.b64encode(audio).decode("ascii")
                    await channel.send_json_async({"event": "audio", "info": {"data_b64": encoded}})
            await channel.send_json_async({"event": "complete", "info": {}})
        except (RuntimeError, WebSocketDisconnect):
            return
        except SpeechRequestError as exc:
            await channel.fail_async(
                code="synthesis_failed",
                message=str(exc) or "Speech synthesis failed",
                close_code=1011,
            )
            return


@router_tts.get(
    "/{config_uuid}/",
    summary="Get a tts config by ID for the authenticated user.",
    response_model=schemas.UserTTSSchema,
    operation_id="get_tts_config",
)
async def get_tts_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedReadableFilterSet(
        models.UserTTS.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserTTS, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="TTS config not found")
    return config.to_schema()


@router_tts.patch(
    "/{config_uuid}/",
    summary="Update a tts config by ID for the authenticated user.",
    response_model=schemas.UserTTSSchema,
    operation_id="update_tts_config",
)
async def update_tts_config_async(
    config_uuid: str,
    config_update: create_partial_model(schemas.UserTTSSchema),  # type: ignore[valid-type]
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedEditableFilterSet(
        models.UserTTS.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserTTS, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="TTS config not found")
    config = await utils.update_tts_config_async(
        session, config, config_update, updated_user_uuid=user.uuid
    )
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_tts.delete(
    "/{config_uuid}/",
    summary="Delete a tts config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="delete_tts_config",
)
async def delete_tts_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    filters = UserScopedEditableFilterSet(
        models.UserTTS.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserTTS, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="TTS config not found")
    await session.delete(config)
    await session.commit()


# ============================================================================
# Agent Config Endpoints
# ============================================================================

router_agents = APIRouter(prefix="/configs/agents", tags=["agent_configs"])


@router_agents.post(
    "/",
    summary="Create a new agent config for the authenticated user.",
    response_model=schemas.UserAgentSchema,
    status_code=status.HTTP_201_CREATED,
    operation_id="create_agent_config",
)
async def create_agent_config_async(
    config_create: schemas.UserAgentSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await utils.create_agent_config_async(
        session,
        None if user.is_superuser else user.uuid,
        config_create,
        updated_user_uuid=user.uuid,
    )
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_agents.get(
    "/",
    summary="List all agent configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserAgentSchema],
    operation_id="list_agent_configs",
)
async def list_agent_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse:
    filters = UserScopedReadableFilterSet(
        models.UserAgent.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    configs = await utils.list_user_scoped_async(
        session, models.UserAgent, filters=filters, skip=skip, limit=limit
    )
    total = await utils.count_user_scoped_async(session, models.UserAgent, filters=filters)
    return PaginatedResponse(
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_agents.get(
    "/{config_uuid}/",
    summary="Get a agent config by ID for the authenticated user.",
    response_model=schemas.UserAgentSchema,
    operation_id="get_agent_config",
)
async def get_agent_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedReadableFilterSet(
        models.UserAgent.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserAgent, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent config not found")
    return config.to_schema()


@router_agents.patch(
    "/{config_uuid}/",
    summary="Update a agent config by ID for the authenticated user.",
    response_model=schemas.UserAgentSchema,
    operation_id="update_agent_config",
)
async def update_agent_config_async(
    config_uuid: str,
    config_update: create_partial_model(schemas.UserAgentSchema),  # type: ignore[valid-type]
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedEditableFilterSet(
        models.UserAgent.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserAgent, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent config not found")
    _reject_frozen_agent_update(config, config_update, user)
    config = await utils.update_agent_config_async(
        session, config, config_update, updated_user_uuid=user.uuid
    )
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_agents.delete(
    "/{config_uuid}/",
    summary="Delete a agent config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="delete_agent_config",
)
async def delete_agent_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    filters = UserScopedEditableFilterSet(
        models.UserAgent.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserAgent, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent config not found")
    _reject_frozen_agent_delete(config, user)
    await session.delete(config)
    await session.commit()


# ============================================================================
# Tool Config Endpoints
# ============================================================================

router_tools = APIRouter(prefix="/configs/tools", tags=["tool_configs"])


@router_tools.post(
    "/index/",
    summary="Index tool for the authenticated user.",
    status_code=status.HTTP_200_OK,
)
async def index_tool_async(
    user: IUser = Depends(get_authenticated_user_async),
    config_provider: IUserConfigProvider = Depends(get_config_provider_async),
):
    agent_tools = await create_tool_retriever_async(
        tool_backend=config_provider.get_tool_backend(),
        tool_config_repository=config_provider.get_tool_repository(user_uuid=user.uuid),
        embedding_backend=config_provider.get_embedding_backend(),
        embedding_config_repository=config_provider.get_embedding_repository(user_uuid=user.uuid),
        space_id=user.uuid,
    )
    await agent_tools.index_tools_async()


@router_tools.post(
    "/{config_uuid}/probe/",
    summary="Probe tool for the authenticated user.",
    status_code=status.HTTP_200_OK,
)
async def probe_tool_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
    config_provider: IUserConfigProvider = Depends(get_config_provider_async),
) -> schemas.UserToolProbeSchema:
    filters = UserScopedReadableFilterSet(
        models.UserTool.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserTool, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool config not found",
        )
    tool_schema = config.to_schema()
    await session.close()

    tool_backend = config_provider.get_tool_backend()
    tool_bundle = tool_backend.create_tool_bundle(tool_schema)
    tool_context = tool_bundle.setup()
    async with tool_context as tools:
        tool_names = [tool.name for tool in tools]

    return schemas.UserToolProbeSchema(tool_names=tool_names)


@router_tools.post(
    "/",
    summary="Create a new tool config for the authenticated user.",
    response_model=schemas.UserToolSchema,
    status_code=status.HTTP_201_CREATED,
    operation_id="create_tool_config",
)
async def create_tool_config_async(
    config_create: schemas.UserToolSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await utils.create_tool_config_async(
        session,
        None if user.is_superuser else user.uuid,
        config_create,
        updated_user_uuid=user.uuid,
    )
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_tools.get(
    "/",
    summary="List all tool configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserToolSchema],
    operation_id="list_tool_configs",
)
async def list_tool_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse:
    filters = UserScopedReadableFilterSet(
        models.UserTool.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    configs = await utils.list_user_scoped_async(
        session, models.UserTool, filters=filters, skip=skip, limit=limit
    )
    total = await utils.count_user_scoped_async(session, models.UserTool, filters=filters)
    return PaginatedResponse(
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_tools.get(
    "/{config_uuid}/",
    summary="Get a tool config by ID for the authenticated user.",
    response_model=schemas.UserToolSchema,
    operation_id="get_tool_config",
)
async def get_tool_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedReadableFilterSet(
        models.UserTool.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserTool, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool config not found")
    return config.to_schema()


@router_tools.patch(
    "/{config_uuid}/",
    summary="Update a tool config by ID for the authenticated user.",
    response_model=schemas.UserToolSchema,
    operation_id="update_tool_config",
)
async def update_tool_config_async(
    config_uuid: str,
    config_update: create_partial_model(schemas.UserToolSchema),  # type: ignore[valid-type]
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedEditableFilterSet(
        models.UserTool.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserTool, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool config not found")
    config = await utils.update_tool_config_async(
        session, config, config_update, updated_user_uuid=user.uuid
    )
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_tools.delete(
    "/{config_uuid}/",
    summary="Delete a tool config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="delete_tool_config",
)
async def delete_tool_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    filters = UserScopedEditableFilterSet(
        models.UserTool.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserTool, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool config not found")
    await session.delete(config)
    await session.commit()


# ============================================================================
# Skill Config Endpoints
# ============================================================================

router_skills = APIRouter(prefix="/configs/skills", tags=["skill_configs"])


@router_skills.post(
    "/",
    summary="Create a new skill config for the authenticated user.",
    response_model=schemas.UserSkillSchema,
    status_code=status.HTTP_201_CREATED,
    operation_id="create_skill_config",
)
async def create_skill_config_async(
    config_create: schemas.UserSkillSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = await utils.create_skill_config_async(
        session,
        None if user.is_superuser else user.uuid,
        config_create,
        updated_user_uuid=user.uuid,
    )
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_skills.get(
    "/",
    summary="List all skill configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserSkillSchema],
    operation_id="list_skill_configs",
)
async def list_skill_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse:
    filters = UserScopedReadableFilterSet(
        models.UserSkill.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    configs = await utils.list_user_scoped_async(
        session, models.UserSkill, filters=filters, skip=skip, limit=limit
    )
    total = await utils.count_user_scoped_async(session, models.UserSkill, filters=filters)
    return PaginatedResponse(
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_skills.get(
    "/{config_uuid}/",
    summary="Get a skill config by ID for the authenticated user.",
    response_model=schemas.UserSkillSchema,
    operation_id="get_skill_config",
)
async def get_skill_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedReadableFilterSet(
        models.UserSkill.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserSkill, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill config not found")
    return config.to_schema()


@router_skills.patch(
    "/{config_uuid}/",
    summary="Update a skill config by ID for the authenticated user.",
    response_model=schemas.UserSkillSchema,
    operation_id="update_skill_config",
)
async def update_skill_config_async(
    config_uuid: str,
    config_update: create_partial_model(schemas.UserSkillSchema),  # type: ignore[valid-type]
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedEditableFilterSet(
        models.UserSkill.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserSkill, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill config not found")
    config = await utils.update_skill_config_async(
        session, config, config_update, updated_user_uuid=user.uuid
    )
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_skills.delete(
    "/{config_uuid}/",
    summary="Delete a skill config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="delete_skill_config",
)
async def delete_skill_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    filters = UserScopedEditableFilterSet(
        models.UserSkill.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserSkill, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill config not found")
    await session.delete(config)
    await session.commit()


# ============================================================================
# Question Config Endpoints
# ============================================================================

router_questions = APIRouter(prefix="/configs/questions", tags=["question_configs"])


@router_questions.post(
    "/",
    summary="Create a new question config for the authenticated user.",
    response_model=schemas.UserQuestionSchema,
    status_code=status.HTTP_201_CREATED,
    operation_id="create_question_config",
)
async def create_question_config_async(
    config_create: schemas.UserQuestionSchema,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    config = models.UserQuestion(
        id=config_create.id,
        user_uuid=None if user.is_superuser else user.uuid,
        question=config_create.question,
        answer=config_create.answer,
        is_active=config_create.is_active if hasattr(config_create, "is_active") else False,
        updated_at=datetime.now(timezone.utc),
        updated_user_uuid=user.uuid,
    )
    session.add(config)
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_questions.get(
    "/",
    summary="List all question configs for the authenticated user.",
    response_model=PaginatedResponse[schemas.UserQuestionSchema],
    operation_id="list_question_configs",
)
async def list_question_configs_async(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    is_active: bool | None = Query(None),
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> PaginatedResponse:
    filters = QuestionFilterSet(user.uuid, is_superuser=user.is_superuser)
    filters.parse(is_active=is_active)
    configs = await utils.list_user_scoped_async(
        session,
        models.UserQuestion,
        filters=filters,
        skip=skip,
        limit=limit,
    )
    total = await utils.count_user_scoped_async(session, models.UserQuestion, filters=filters)
    return PaginatedResponse(
        total=total,
        results=[config.to_schema() for config in configs],
    )


@router_questions.get(
    "/{config_uuid}/",
    summary="Get a question config by ID for the authenticated user.",
    response_model=schemas.UserQuestionSchema,
    operation_id="get_question_config",
)
async def get_question_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedReadableFilterSet(
        models.UserQuestion.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserQuestion, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Question config not found"
        )
    return config.to_schema()


@router_questions.patch(
    "/{config_uuid}/",
    summary="Update a question config by ID for the authenticated user.",
    response_model=schemas.UserQuestionSchema,
    operation_id="update_question_config",
)
async def update_question_config_async(
    config_uuid: str,
    config_update: create_partial_model(schemas.UserQuestionSchema),  # type: ignore[valid-type]
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
):
    filters = UserScopedEditableFilterSet(
        models.UserQuestion.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserQuestion, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Question config not found"
        )
    fields_set: set[str] = getattr(config_update, "model_fields_set", set())
    if "question" in fields_set and config_update.question is not None:
        config.question = config_update.question
    if "answer" in fields_set:
        config.answer = config_update.answer
    if "is_active" in fields_set and config_update.is_active is not None:
        config.is_active = config_update.is_active
    config.updated_at = datetime.now(timezone.utc)
    config.updated_user_uuid = user.uuid
    session.add(config)
    await session.commit()
    await session.refresh(config)
    return config.to_schema()


@router_questions.delete(
    "/{config_uuid}/",
    summary="Delete a question config by ID for the authenticated user.",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="delete_question_config",
)
async def delete_question_config_async(
    config_uuid: str,
    user: IUser = Depends(get_authenticated_user_async),
    session: AsyncSession = Depends(get_db_session_async),
) -> None:
    filters = UserScopedEditableFilterSet(
        models.UserQuestion.user_uuid, user.uuid, is_superuser=user.is_superuser
    )
    config = await utils.get_user_scoped_async(
        session, models.UserQuestion, filters=filters, config_uuid=config_uuid
    )
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Question config not found"
        )
    await session.delete(config)
    await session.commit()
