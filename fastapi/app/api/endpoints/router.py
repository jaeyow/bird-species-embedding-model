import json
from fastapi import APIRouter, File, UploadFile

from app.services.generator import EmbeddingGeneratorService
import lancedb

router = APIRouter()

@router.post(
    "/get_similar_birds",
    tags=["get_similar_birds"],
    summary="Get similar birds")
async def get_similar_birds(file: UploadFile = File(...), limit: int=6, metric: str="cosine"):
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
    results["label"] = results["image_path"].apply(lambda x: x.split("/")[-2])
    results["similarity"] = 1 - results["_distance"]
    results = results.drop(columns=["_distance"]) # Drop the distance column, just need the similarity

    results_list = results.to_dict(orient="records")

    return results_list


