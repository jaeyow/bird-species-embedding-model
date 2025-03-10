import numpy as np
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


class GeneratorService:
    def __init__(
        self,
        model_path: str = "./model/resnet50d.ra2_in1k_fine_tune_51_classes_2024-10-06_12-01-37.pth",
    ):
        self.resnet_model = self.load_model(model_path)

    def load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load the ResNet model
        """
        model = timm.create_model("resnet50d", pretrained=False, num_classes=51)
        model.reset_classifier(0)

        checkpoint = torch.load(model_path)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    def preprocess_image(
        self, image: Image.Image, device: str, target_size: tuple = (224, 224)
    ) -> torch.Tensor:
        """
        Preprocess the input image
        """
        transform = transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.resnet_model.default_cfg["mean"],
                    std=self.resnet_model.default_cfg["std"],
                ),
            ]
        )

        input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        return input_tensor

    def generate_embedding(self, img_path: str) -> np.ndarray:
        """
        Generate embeddings for the given input image
        """
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        input_tensor = self.preprocess_image(Image.open(img_path), device)
        self.resnet_model.to(device)
        self.resnet_model.eval()

        with torch.no_grad():
            embedding = self.resnet_model(input_tensor)

        embedding = embedding.cpu().numpy().flatten()

        return embedding
