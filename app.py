from fastapi import FastAPI
from routes.face.main_routes import face_router
app = FastAPI()

app.include_router(face_router, prefix="/v1/face", tags=["face"])

@app.get("/")
def read_root():
    return {"Message": "Ok"}