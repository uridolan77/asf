from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from .models import UserCreate, UserLogin, UserOut, Token
from .security import authenticate_user, create_access_token, get_current_user

router = APIRouter()

@router.post("/register", response_model=UserOut)
def register(user: UserCreate):
    # TODO: Implement user creation logic (DB insert, hash password)
    raise NotImplementedError("User registration not implemented.")

@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # TODO: Implement authentication logic
    raise NotImplementedError("User login not implemented.")

@router.get("/me", response_model=UserOut)
def get_me(current_user: UserOut = Depends(get_current_user)):
    return current_user
