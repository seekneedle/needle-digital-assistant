from fastapi import APIRouter, Depends, Request
import traceback
from utils.config import config
from pydantic import BaseModel
from server.auth import check_permission

from utils.log import log
from services.trainer import trainer, TrainerRequest
from services.trainer_score import trainer_score, TrainerScoreRequest
from services.task import get_task_id, AssistantRequest, AssistantTaskResponse
from services.text import get_text_message
from fastapi.responses import StreamingResponse

from server.response import SuccessResponse, FailResponse

store_router = APIRouter(prefix='/assistant', dependencies=[Depends(check_permission)])


# 1. trainer聊天
@store_router.post('/trainer')
async def assistant_trainer(request: Request, trainer_request: TrainerRequest):
    log.info(f'incoming request: {trainer_request}')
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
    log.info(f'getTaskId: {request}')
    task_id = get_task_id(request)
    log.info(f"/getTaskId, task_id: {task_id}, request: {request}")
    return SuccessResponse(data=AssistantTaskResponse(taskId=task_id))


class TaskRequest(BaseModel):
    taskId: str


# 获取文本流式输出
@store_router.post('/textMessage')
async def textMessage(request: Request, task_request: TaskRequest):
    try:
        async def event_stream():
            buffer = []
            try:
                async for event in get_text_message(task_request.taskId):
                    if await request.is_disconnected():
                        break
                    buffer.append(event.strip())
                    yield event
            finally:
                merged_content = "".join(buffer)  # 合并所有事件内容
                log.info(f"/textMessage, task_id: {task_request.taskId}, text message: {merged_content}")

        return StreamingResponse(event_stream(), media_type='text/event-stream')
    except Exception as e:
        trace_info = traceback.format_exc()
        log.error(
            f'Exception for /textMessage, task_id: {task_request.taskId}, e: {e}, trace: {trace_info}')
        return FailResponse(error=str(e))
