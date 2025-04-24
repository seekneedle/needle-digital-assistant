from data.database import TableModel
from sqlalchemy import Column, String, Integer


class AssistantEntity(TableModel):
    task_id = Column(String)
    role = Column(String)
    messages = Column(String)