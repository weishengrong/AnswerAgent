from pydantic import BaseModel, Field


class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=50)
    nickname: str = Field(..., min_length=1, max_length=100)


class UserLogin(BaseModel):
    username: str = Field(...)
    password: str = Field(...)


class UserResponse(BaseModel):
    id: int
    username: str
    nickname: str | None

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
