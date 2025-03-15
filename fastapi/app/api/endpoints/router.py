from typing import List

from pydantic import BaseModel
from fastapi import APIRouter, File, UploadFile, Query

from app.services.generator import EmbeddingGeneratorService
import lancedb

router = APIRouter()

class BirdSimilarityResult(BaseModel):
    image_path: str
    label: str
    similarity: float
    
@router.post("/get_similar_birds", response_model=List[BirdSimilarityResult])
async def get_similar_birds(
    file: UploadFile = File(...),
    limit: int = Query(default=6),
    metric: str = Query(default="cosine")
):
    content = await file.read()
    embedding_service = EmbeddingGeneratorService()
    unknown_bird = embedding_service.generate_embedding_from_bytes(content)
    
    db = lancedb.connect("/app/app/lancedb")
    table = db.open_table("embeddings")

    results = table \
        .search(unknown_bird) \
        .metric(metric) \
        .limit(limit) \
        .to_pandas()
    
    results = results[["image_path", "_distance"]]
    # Remove the "../kaggle_data/all_birds/" prefix from image paths
    results["image_path"] = results["image_path"].apply(lambda x: x.split("/")[-2] + "/" + x.split("/")[-1])
    results["label"] = results["image_path"].apply(lambda x: x.split("/")[0])
    results["similarity"] = 1 - results["_distance"]
    results = results.drop(columns=["_distance"]) # Drop the distance column, just need the similarity

    results_list = [BirdSimilarityResult(**result) for result in results.to_dict(orient="records")]

    return results_list

@router.get("/test-cors")
async def test_cors():
    return {"message": "CORS is working"}


