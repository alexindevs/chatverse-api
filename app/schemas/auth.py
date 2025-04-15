from pydantic import BaseModel, EmailStr

# Schema for user signup
class UserCreate(BaseModel):
    email: EmailStr
    name: str
    username: str
    password: str

# Schema for user login
class UserLogin(BaseModel):
    email: EmailStr
    password: str

# Schema for token response
class Token(BaseModel):
    access_token: str
    token_type: str