import json
import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline, VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

print(f"GPU is available: {torch.cuda.is_available()}")

config = json.load(open("config.json"))

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

# Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=config['HUGGINGFACE_API_KEY'])
pipe = pipe.to(device)

# GPT-2
generator = pipeline('text-generation', model="gpt2")

# Image Captioning
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.to(device)