from passlib.context import CryptContext
from fastapi import Form


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_form_signupdata(
        username: str = Form(...),
        email: str = Form(...),
        password: str = Form(...)
    ) -> dict:
    return {
        "username": username,
        "email": email,
        "password": password
    }

def get_form_logindata(
        email: str = Form(...),
        password: str = Form(...)
    ) -> dict:
    return {
        "email": email,
        "password": password
    }
