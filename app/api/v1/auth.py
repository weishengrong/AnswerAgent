"""认证路由"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.models import User
from app.auth.schemas import UserRegister, UserLogin, TokenResponse, UserResponse
from app.auth.service import (
    create_access_token_for_user,
    get_db,
    get_current_user_from_token,
    get_user_by_username,
    create_user,
    authenticate_user,
)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

router = APIRouter(prefix="/auth", tags=["auth"])


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    return await get_current_user_from_token(token, db)


@router.post("/register", response_model=TokenResponse)
async def register(user_data: UserRegister, db: AsyncSession = Depends(get_db)):
    existing_user = await get_user_by_username(db, user_data.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="用户名已存在")

    new_user = await create_user(
        db,
        username=user_data.username,
        password=user_data.password,
        nickname=user_data.nickname
    )

    access_token = create_access_token_for_user(new_user.id, new_user.username)

    return TokenResponse(
        access_token=access_token,
        user=UserResponse(
            id=new_user.id,
            username=new_user.username,
            nickname=new_user.nickname
        )
    )


@router.post("/login", response_model=TokenResponse)
async def login(user_data: UserLogin, db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, user_data.username, user_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    access_token = create_access_token_for_user(user.id, user.username)

    return TokenResponse(
        access_token=access_token,
        user=UserResponse(
            id=user.id,
            username=user.username,
            nickname=user.nickname
        )
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="未登录")
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        nickname=current_user.nickname
    )
