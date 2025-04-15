from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select  # Import select
from ..models.user import User
from ..schemas.auth import UserCreate, UserLogin, Token
from ..utils.auth import hash_password, verify_password, create_access_token, retrieve_token_data
from ..config.dependencies import get_db
from typing import Dict, Any
from pydantic import BaseModel

class UserTokenData(BaseModel):
    user_id: int
    name: str
    email: str

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

router = APIRouter()
@router.post("/signup", response_model=Token)
async def signup(user: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == user.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = hash_password(user.password)
    new_user = User(email=user.email, hashed_password=hashed_password, name=user.name, username=user.username)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # Generate token
    access_token = create_access_token(data={
        "user_id": new_user.id,
        "name": user.name,
        "email": user.email
    })
    return {"message": "Signup Successful", "access_token": access_token, "token_type": "bearer"}


# Login route
@router.post("/login", response_model=Token)
async def login(form_data: UserLogin, db: AsyncSession = Depends(get_db)):
    # Get user from database
    result = await db.execute(select(User).where(User.email == form_data.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    # Generate token
    access_token = create_access_token(data={
        "user_id": user.id,
        "name": user.name,
        "email": user.email
    })
    return {"message":"Login Successful", "access_token": access_token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme)) -> UserTokenData:
    """
    Decode the JWT token and return its payload (the data field).
    """
    token_data = retrieve_token_data(token)
    return UserTokenData(**token_data)

@router.get("/me", response_model=UserTokenData)
async def get_me(current_user: UserTokenData = Depends(get_current_user)):
    """
    Returns the currently authenticated user's information.
    """
    return current_user
