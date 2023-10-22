import numpy as np
from tqdm import tqdm
from torch import autocast
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from diffusers import DiffusionPipeline
import torch

class Diffs:
    def __init__(self, device="cuda", *, hf_token, model_type="CompVis/stable-diffusion-v1-4"):
        self.model = SentenceTransformer('clip-ViT-B-32')
        self.device = device
        self.pipe = DiffusionPipeline.from_pretrained(
            model_type,
            use_auth_token=hf_token,
            torch_dtype=torch.float16
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def generate_images(self, query, num_images=10):
        images = []
        pbar = tqdm(total=num_images, position=0)

        pbar.set_description(f"Generating query {query}")

        for _ in range(0, num_images):

            with autocast("cuda"):
                image = self.pipeline(query).images[0]

            images.append(image)
            pbar.update(1)
        pbar.close()
        return images

    def generate_image_embedding(self, query, num_images=10, return_image=False):

        embeddings = []
        images = self.generate_images(query, num_images)

        for img in images:
            embeddings.append(self.model.encode(img))

        embeddings = np.mean(embeddings, axis=0)

        if return_image:
            return embeddings, images
        else:
            return embeddings

    def generate_sentence_embedding(self, query):
        # not the smartest way
        return self.model.encode([query])[0]

