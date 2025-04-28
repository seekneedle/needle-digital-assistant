from fastapi import APIRouter, Depends, Request, WebSocket
import traceback
from utils.config import config
from pydantic import BaseModel
from server.auth import check_permission
import sys

from utils.log import log
from services.trainer import trainer, TrainerRequest
from services.trainer_scoring import trainer_score, TrainerScoreRequest
from services.task import get_task_id, AssistantRequest, AssistantTaskResponse
from services.text import get_text_message
from services.asr_proxy import proxy_websocket
from services.tts_proxy import to_speech, TtsRequest

from fastapi.responses import StreamingResponse

from server.response import SuccessResponse, FailResponse

store_router = APIRouter(prefix='/assistant', dependencies=[Depends(check_permission)])
ws_router = APIRouter(prefix='/assist')

# 1. trainer聊天
@store_router.post('/trainer')
async def assistant_trainer(request: Request, trainer_request: TrainerRequest):
    log.info(f'/trainer incoming request: {trainer_request}')
    try:
        async def event_stream():
            async for event in trainer(trainer_request):
                if await request.is_disconnected():
                    break
                yield event

        return StreamingResponse(event_stream(), media_type='text/event-stream')
    except Exception as e:
        trace_info = traceback.format_exc()
        log.error(f'Exception for /assistant/trainer, request: {request}, e: {e}, trace: {trace_info}')
        return FailResponse(error=str(e))

# 2. trainer评分
@store_router.post('/trainer_score')
async def assistant_trainer_score(request: Request, trainer_score_request: TrainerScoreRequest):
    try:
        async def event_stream():
            async for event in trainer_score(trainer_score_request):
                if await request.is_disconnected():
                    break
                yield event

        return StreamingResponse(event_stream(), media_type='text/event-stream')
    except Exception as e:
        trace_info = traceback.format_exc()
        log.error(f'Exception for /assistant/trainer_score, request: {request}, e: {e}, trace: {trace_info}')
        return FailResponse(error=str(e))


# 获取task_id
@store_router.post('/getTaskId')
async def getTaskId(request: AssistantRequest):
    log.info(f'/getTaskId incoming request: {request}')
    task_id = get_task_id(request)
    log.info(f"/getTaskId, task_id: {task_id}, request: {request}")
    return SuccessResponse(data=AssistantTaskResponse(taskId=task_id))


class TaskRequest(BaseModel):
    taskId: str


# 获取文本流式输出
@store_router.post('/textMessage')
async def textMessage(request: Request, task_request: TaskRequest):
    log.info(f'/textMessage incoming request: {task_request}')
    try:
        async def event_stream():
            buffer = []
            try:
                async for event in get_text_message(task_request.taskId):
                    if await request.is_disconnected():
                        break
                    buffer.append(event.strip())
                    yield f"data: {event}\n\n"
                    sys.stdout.flush()
            finally:
                merged_content = "".join(buffer)  # 合并所有事件内容
                log.info(f"/textMessage, task_id: {task_request.taskId}, text message: {merged_content}")

        return StreamingResponse(event_stream(), media_type='text/event-stream', headers={
            "Cache-Control": "no-cache",        # 禁用客户端缓存
            "X-Accel-Buffering": "no",          # 禁用Nginx等代理缓冲
            "Connection": "keep-alive"          # 保持长连接
        })
    except Exception as e:
        trace_info = traceback.format_exc()
        log.error(
            f'Exception for /textMessage, task_id: {task_request.taskId}, e: {e}, trace: {trace_info}')
        return FailResponse(error=str(e))


# 2. trainer 评分，新接口
@store_router.post('/trainerScore')
async def assistant_trainer_score_new(request: Request, trainer_score_request: TrainerScoreRequest):
    log.info(f'/trainerScore incoming request: {trainer_score_request}')
    res = await trainer_score(trainer_score_request)
    log.info(f'/trainerScore response: {res}')
    return res

@store_router.post('/tts')
async def tts_endpoint(request: Request, tts_request: TtsRequest):
    log.info(f'/tts incoming request: {tts_request}')
    res = await to_speech(tts_request)
    # log.info(f'/tts response: {res}')
    return res

# asr 语音转文字，转发
@ws_router.websocket('/ws_asr')
async def websocket_endpoint(websocket: WebSocket):
    log.info(f'/ws_asr incoming request from {websocket.client}')
    try:
        await websocket.accept()
        # log.info('WebSocket connection accepted successfully')
        await proxy_websocket(websocket)
    except Exception as e:
        log.error(f'WebSocket connection error: {type(e).__name__} - {e}')
        try:
            await websocket.close(code=1011)  # Internal server error
        except Exception:
            pass
