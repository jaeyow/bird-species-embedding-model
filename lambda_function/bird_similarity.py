import json
import timm
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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


def cosine_similarity_(embedding1, embedding2):
    """
    Create a cosine similarity function that takes two embeddings and returns the cosine similarity between them.
    """
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]


def generate_embedding(image_path, model):
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

    image = Image.open(image_path)
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        embedding = model(input_tensor)

    embedding = embedding.cpu().numpy().flatten()

    return embedding


def handler(event, context):
    """
    Endpoint to classify bird species using the KNN model
    """
    message = "Hello from the bird_similarity FastAPI service"
    print(f"Event: {event}")
    
    embedding = generate_embedding("4.jpg", resnet_model)

    return {
        "statusCode": 200,
        "message": message,
        "embedding": embedding.tolist()[:5],
    }
