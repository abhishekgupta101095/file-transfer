import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/try")
def trying():
    return {"message":"I am trying"}
#@app.post("/hello")
#async def hello():
#    return "welcome"

if __name__ == "__main__":
    uvicorn.run("test_api:app", host="127.0.0.1", port=8090)