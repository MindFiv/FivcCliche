from pydantic import BaseModel


class UserQuestionListQuery(BaseModel):
    is_active: bool | None = None
