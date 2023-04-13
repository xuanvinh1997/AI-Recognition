from fastapi import APIRouter
from pipelines.face import FacePipeline
face_router = APIRouter()
face_pipeline = FacePipeline()

@face_router.get("/health")
def health():
    return {"Message": "Ok"}
