from pydantic import BaseModel
from typing import List, Dict
from data.assistant import AssistantEntity
import uuid
import json


class Role(BaseModel):
    level: str  # 必填字段
    personality: str = "友好型"  # 可选字段
    personality_example: str = '"您详细说说这个行程吧，我慢慢听"'  # 可选字段
    demand: str = '基础需求型'  # 可选字段
    demand_example: str = '"这个行程价格是多少？"'  # 可选字段
    background: str = '普通上班族'  # 可选字段
    background_example: str = '"我是刚毕业的大学生"'  # 可选字段
    region: str = '本地化需求'  # 可选字段
    region_example: str = '"行程中有没有故宫周边景点？"'  # 可选字段


class AssistantRequest(BaseModel):
    role: Role
    messages: List[object]


class AssistantTaskResponse(BaseModel):
    taskId: str


def get_task_id(request: AssistantRequest):
    task_id = str(uuid.uuid4())
    AssistantEntity.create(task_id=task_id,
                        role=request.role.model_dump_json(indent=4),
                        messages=json.dumps(request.messages, ensure_ascii=False, indent=4))
    return task_id