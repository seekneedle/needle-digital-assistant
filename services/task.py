from pydantic import BaseModel
from typing import List, Dict
from data.assistant import AssistantEntity
import uuid
import json


class AssistantRequest(BaseModel):
    role: str
    messages: List[object]


class AssistantTaskResponse(BaseModel):
    taskId: str


def get_task_id(request: AssistantRequest):
    task_id = str(uuid.uuid4())
    AssistantEntity.create(task_id=task_id,
                        role=request.role,
                        messages=json.dumps(request.messages, ensure_ascii=False, indent=4))
    return task_id