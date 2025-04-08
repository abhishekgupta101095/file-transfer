from fastapi import FastAPI
from db import models
from db.database import engine
from routers import organizations
from routers import users
app = FastAPI()

app.include_router(organizations.router)
app.include_router(users.router)
models.Base.metadata.create_all(bind=engine)