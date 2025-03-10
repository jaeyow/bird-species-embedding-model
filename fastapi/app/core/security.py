from fastapi import Security, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext

# Security scheme for OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_current_user(token: str = Security(oauth2_scheme)):
    # Logic to decode the token and retrieve the user
    raise HTTPException(status_code=401, detail="Invalid authentication credentials")

def get_current_active_user(current_user: str = Security(get_current_user)):
    # Logic to check if the user is active
    return current_user