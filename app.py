from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
from jose import jwt
import logging
import os
from datetime import datetime, timedelta
from database import users_collection
from bson import ObjectId
from passlib.context import CryptContext



load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")

# ------------------- Password Hashing ---------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# ------------------- Load Env and Setup ---------------------
load_dotenv()
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
template = Jinja2Templates(directory="template")

# ------------------- Routes ----------------------------
@app.get("/debug/users")
async def debug_users():
    users = []
    async for user in users_collection.find():
        users.append(user)
    return users


@app.get("/get-user/{username}")
async def get_user(username: str):
    user = await users_collection.find_one({"username": username})
    if user:
        user["_id"] = str(user["_id"])
        return JSONResponse(content=user)
    return JSONResponse(content={"error": "User not found"}, status_code=404)

@app.get("/", response_class=HTMLResponse)
async def redirect_to_login(request: Request):
    return RedirectResponse("/login")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/login")
    return template.TemplateResponse("index.html", {"request": request, "user": user})

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return template.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def show_register_page(request: Request):
    return template.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register_user(username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    existing_user = await users_collection.find_one({"username": username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_pw = hash_password(password)

    new_user = {
        "username": username,
        "email": email,
        "hashed_password": hashed_pw,
        "created_at": datetime.utcnow()
    }

    await users_collection.insert_one(new_user)
    return RedirectResponse("/login", status_code=302)

@app.post("/login")
async def login_post(request: Request):
    form = await request.form()
    username = form.get("username")
    password = form.get("password")

    user = await users_collection.find_one({"username": username})
    if not user or not verify_password(password, user["hashed_password"]):
        return template.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid credentials"
        })

    
    request.session["user"] = {"username": user["username"]}
    return RedirectResponse("/dashboard", status_code=302)

@app.get("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse("/login")

@app.post("/api/login")
async def api_login(username: str = Form(...), password: str = Form(...)):
    user = await users_collection.find_one({"username": username})
    if user and verify_password(password, user["hashed_password"]):
        access_token = jwt.encode({
            "sub": username,
            "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        }, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

        return JSONResponse(status_code=200, content={
            "access_token": access_token,
            "message": "Login successful!"
        })
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

@app.post("/upload_and_query")
async def upload_and_query(
    image: UploadFile = File(...),
    query: str = Form(...)
):
    try:
        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="Empty file")

        encoded_image = base64.b64encode(image_content).decode("utf-8")

        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ]

        def make_api_request(model):
            response = requests.post(
                GROQ_API_URL,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 1000
                },
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            return response

        llama_response = make_api_request("meta-llama/llama-4-scout-17b-16e-instruct")
        llava_response = make_api_request("meta-llama/llama-4-scout-17b-16e-instruct")

        responses = {}
        for model, response in [("llama", llama_response), ("llava", llava_response)]:
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                logger.info(f"Response from {model} API: {answer[:100]}...")
                responses[model] = answer
            else:
                logger.error(f"Error from {model} API: {response.status_code} - {response.text}")
                responses[model] = f"Error from {model} API: {response.status_code}"

        return JSONResponse(status_code=200, content=responses)

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# ------------------- Run -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
