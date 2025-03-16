from typing import List
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from pydantic import BaseModel, Field

from app.services.generator import EmbeddingGeneratorService
import lancedb

router = APIRouter(
    prefix="/api/v1",
    tags=["Bird Similarity Search"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)

class BirdSimilarityResult(BaseModel):
    """
    Represents a similar bird result from the search
    """
    image_path: str = Field(
        description="Path to the bird image",
        example="NORTHERN_CARDINAL/001.jpg"
    )
    label: str = Field(
        description="Bird species name",
        example="NORTHERN_CARDINAL"
    )
    similarity: float = Field(
        description="Similarity score between 0 and 1",
        example=0.95,
        ge=0,
        le=1
    )

    class Config:
        schema_extra = {
            "example": {
                "image_path": "NORTHERN_CARDINAL/001.jpg",
                "label": "NORTHERN_CARDINAL",
                "similarity": 0.95
            }
        }

@router.post(
    "/similar-birds",
    response_model=List[BirdSimilarityResult],
    summary="Find similar birds",
    description="Upload a bird image and find similar birds in the vector database using similarity search",
    responses={
        200: {
            "description": "Successfully found similar birds",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "image_path": "NORTHERN_CARDINAL/001.jpg",
                            "label": "NORTHERN_CARDINAL",
                            "similarity": 0.95
                        },
                        {
                            "image_path": "NORTHERN_CARDINAL/002.jpg",
                            "label": "NORTHERN_CARDINAL",
                            "similarity": 0.92
                        }
                    ]
                }
            }
        },
        400: {
            "description": "Invalid input",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid image format"}
                }
            }
        }
    }
)
async def get_similar_birds(
    file: UploadFile = File(
        ...,
        description="Bird image file to search for similar birds"
    ),
    limit: int = Query(
        default=6,
        ge=1,
        le=20,
        description="Number of similar birds to return"
    ),
    metric: str = Query(
        default="cosine",
        description="Similarity metric to use",
        enum=["cosine"]
    )
) -> List[BirdSimilarityResult]:
    """
    Upload a bird image and find similar birds in the database.

    The endpoint will:
    1. Process the uploaded image
    2. Generate an embedding vector using a fine-tuned ResNet model
    3. Search for similar birds in the vector database
    4. Return a list of similar birds with their similarity scores

    Args:
        file: Image file (JPG/PNG) of a bird
        limit: Number of similar birds to return (1-20)
        metric: Similarity metric to use (cosine, more metrics coming soon)

    Returns:
        List of similar birds with their image paths and similarity scores
    
    Raises:
        HTTPException: If the image format is invalid or processing fails
    """
    try:
        contents = await file.read()
        embedding_service = EmbeddingGeneratorService()
        unknown_bird = await embedding_service.generate_embedding_from_bytes(contents)
        
        db = lancedb.connect("/app/app/lancedb")
        table = db.open_table("embeddings")

        results = table \
            .search(unknown_bird) \
            .metric(metric) \
            .limit(limit) \
            .to_pandas()
        
        results = results[["image_path", "_distance"]]
        results["image_path"] = results["image_path"].apply(lambda x: x.split("/")[-2] + "/" + x.split("/")[-1])
        results["label"] = results["image_path"].apply(lambda x: x.split("/")[0])
        results["similarity"] = 1 - results["_distance"]
        results = results.drop(columns=["_distance"])

        return [BirdSimilarityResult(**result) for result in results.to_dict(orient="records")]

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )



