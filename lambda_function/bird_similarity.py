import base64
import json
import os
import numpy as np
import timm
import torch
from io import BytesIO
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms

def load_model():
    """
    Load the ResNet model and the KNN model.
    """
    
    resnet_model = timm.create_model("resnet50d", pretrained=False, num_classes=51)
    resnet_model.reset_classifier(0)

    checkpoint = torch.load(
        "./model/resnet50d.ra2_in1k_fine_tune_51_classes_2024-10-06_12-01-37.pth"
    )

    if "model_state_dict" in checkpoint:
        resnet_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        resnet_model.load_state_dict(checkpoint)

    resnet_model.eval()
    return resnet_model

def cosine_similarity_(embedding1, embedding2):
    """
    Create a cosine similarity function that takes two embeddings and returns the cosine similarity between them.
    """
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]


def generate_embedding(image, model):
    """
    Generate embeddings for the input image.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=model.default_cfg["mean"], std=model.default_cfg["std"]
            ),
        ]
    )

    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        embedding = model(input_tensor)

    embedding = embedding.cpu().numpy().flatten()

    return embedding


def get_all_embeddings(folder_path, embedding_model):
    """
    Get all embeddings of images in the folder using the embedding model
    """
    
    embeddings = []
    image_paths = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            # if file == "1.jpg":
            image_path = os.path.join(root, file)
            # print(f"Processing {image_path}")
            image = Image.open(image_path)
            embedding = generate_embedding(image, embedding_model)
            embeddings.append(embedding)
            image_paths.append(image_path)

    return np.array(embeddings), image_paths

# all_embeddings = get_all_embeddings(profiles_dir, resnet_model)

# print(f"Vector embeddings for {len(all_embeddings[0])} bird species.")


resnet_model = load_model()

def handler(event, context):
    try:
        print(f"Event: {event}")
        if "body" not in event:
            return {"statusCode": 400, "body": json.dumps({"error": "No file uploaded"})}
        try:
            if event.get("isBase64Encoded", False):
                image_bytes = base64.b64decode(event["body"])
            else:
                image_bytes = event["body"].encode("utf-8")
            image = Image.open(BytesIO(image_bytes))
        except Exception as img_error:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": f"Invalid image data: {str(img_error)}"
                })
            }

        profiles_dir = "./all_birds/"
        all_embeddings = get_all_embeddings(profiles_dir, resnet_model)
        print(f"Vector embeddings saved for {len(all_embeddings[0])} bird species.")

        embedding = generate_embedding(image, resnet_model)

        return {
            "statusCode": 200,
            "body": json.dumps(
            {
                "message": "File received successfully", 
                "embedding": embedding.tolist()[:5],
                "size": len(image_bytes)
            }),
        }
    
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
