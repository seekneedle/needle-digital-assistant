from fastapi import APIRouter, Depends, Request
import traceback

from server.auth import check_permission

from utils.log import log
from services.trainer import trainer, TrainerRequest
from services.trainer_score import trainer_score, TrainerScoreRequest

from fastapi.responses import StreamingResponse

from server.response import SuccessResponse, FailResponse

store_router = APIRouter(prefix='/assistant', dependencies=[Depends(check_permission)])


# 1. trainer聊天
@store_router.post('/trainer')
async def assistant_trainer(request: Request, trainer_request: TrainerRequest):
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