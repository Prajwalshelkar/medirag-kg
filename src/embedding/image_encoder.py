"""
Encodes chest X-ray images into feature vectors using a pretrained
DenseNet from TorchXRayVision (https://github.com/mlmed/torchxrayvision),
a library of models trained specifically on chest X-ray datasets.


 #pip install torchxrayvision torch torchvision

Usage:
    from src.embedding.image_encoder import ImageEncoder
    encoder = ImageEncoder()
    vector = encoder.encode(Path("data/raw/images/1_IM-0001-4001.dcm.png"))
"""

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import torchxrayvision as xrv


class ImageEncoder:
    def __init__(self, weights: str = "densenet121-res224-all", device: str = None):
        """
        weights: which pretrained TorchXRayVision model to use.
                 "densenet121-res224-all" is trained on a mix of public
                 chest X-ray datasets and is a solid general-purpose choice.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = xrv.models.DenseNet(weights=weights)
        self.model.to(self.device)
        self.model.eval()

        # TorchXRayVision expects: grayscale, resized to 224x224,
        # pixel values scaled to roughly [-1024, 1024]
        self.transform = T.Compose(
            [
                T.Grayscale(num_output_channels=1),
                T.Resize((224, 224)),
            ]
        )

    def _load_and_preprocess(self, image_path) -> torch.Tensor:
        img = Image.open(image_path)
        img = self.transform(img)
        arr = np.array(img).astype(np.float32)

        # normalize to torchxrayvision's expected range
        arr = xrv.datasets.normalize(arr, 255)
        arr = arr[None, ...]  # add channel dimension -> (1, H, W)

        tensor = torch.from_numpy(arr).unsqueeze(0)  # add batch dim -> (1, 1, H, W)
        return tensor.to(self.device)

    @torch.no_grad()
    def encode(self, image_path) -> np.ndarray:
        """Returns a 1D feature vector (embedding) for one X-ray image."""
        tensor = self._load_and_preprocess(image_path)
        features = self.model.features(tensor)  # penultimate-layer features
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        vector = pooled.flatten().cpu().numpy()
        return vector

    @torch.no_grad()
    def encode_batch(self, image_paths: list) -> np.ndarray:
        """Encodes multiple images at once. Returns shape (N, feature_dim)."""
        return np.stack([self.encode(p) for p in image_paths])


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # quick manual test: encode the first image found in data/raw/images
    images_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "images"
    candidates = list(images_dir.rglob("*.png"))[:1]

    if not candidates:
        print(f"No images found under {images_dir}. Run the ingestion scripts first.")
        sys.exit(1)

    encoder = ImageEncoder()
    vec = encoder.encode(candidates[0])
    print(f"Encoded {candidates[0].name}")
    print(f"Embedding shape: {vec.shape}")
    print(f"First 10 values: {vec[:10]}")