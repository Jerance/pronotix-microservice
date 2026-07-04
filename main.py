from fastapi import FastAPI
import uvicorn

from app.routes import predict_router

app = FastAPI()


@app.get("/")
async def health_check():
    return {"message": "OK"}


app.include_router(predict_router)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8002, reload=True)
