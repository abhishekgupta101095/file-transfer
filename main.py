from fastapi import FastAPI
from app.db import models
from app.db.database import engine
from app.routers import user_role

app = FastAPI()
app.include_router(user_role.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

models.Base.metadata.create_all(bind=engine)