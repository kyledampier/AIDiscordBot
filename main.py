import discord
from discord.ext import commands
import os
import io
import json
import validators
import requests
from datetime import datetime
from PIL import Image
import torch
from torch import autocast
from transformers import pipeline, set_seed, VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from diffusers import StableDiffusionPipeline

print(f"GPU is available: {torch.cuda.is_available()}")

config = json.load(open("config.json"))

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

# Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=config['HUGGINGFACE_API_KEY'])
pipe = pipe.to(device)

# GPT-2
generator = pipeline('text-generation', model="gpt2")
set_seed(42)

# Image Captioning
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(images):
  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
#   pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='-', intents=intents)

@bot.command()
async def complete(ctx, *args):
    """Completes written prompt using GPT-2"""
    if len(args) == 0:
        await ctx.send("Please follow the format `-complete <prompt>`")
        return

    prompt = " ".join(args)
    output = generator(prompt, max_length=120, num_return_sequences=1)
    await ctx.channel.send(f"Completed phrase for {ctx.author.mention}:\n{output[0]['generated_text']}")

@bot.command()
async def generate(ctx, *args):
    """Generates two images based on your prompt using Stable Diffusion 1.4"""
    if len(args) == 0:
        await ctx.send("Please follow the format `-generate <prompt>`")
        return

    prompt = " ".join(args)
    loading_msg = await ctx.channel.send(f"Generating **{prompt}** for {ctx.author.mention}... (0/2)")

    t = datetime.now().strftime("%Y%m%d-%H%M%S")
    files = []
    scales = [7.5, 8.5]

    for i in range(len(scales)):
        with autocast("cuda"):
            image = pipe(prompt, guidance_scale=scales[i], num_inference_steps=50).images[0]

        file_path = os.path.join("images", f"{t}-{i}.png")
        image.save(file_path)
        files.append(discord.File(file_path))
        await loading_msg.edit(content=f"Generating images for {ctx.author.mention}... ({i+1}/{len(scales)})")

    await ctx.channel.send(f"**{prompt}** (requested by {ctx.author.mention})", files=files)
    await loading_msg.delete()

@bot.command()
async def describe(ctx, *args):
    """Describes the message's attachment or image_url into words"""
    if len(args) > 0:
        if validators.url(args[0]):
            url = args[0]
        else:
            await ctx.send("Error: Invalid url format")
            return
    elif len(ctx.message.attachments) > 0:
        attached_file = ctx.message.attachments[0]
        url = attached_file
    else:
        await ctx.send("No image found. Please follow the format `-describe <attachment|image_url>`")
        return

    try:
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content))
    except:
        await ctx.send("Error: Retrieving image from URL failed")
    else:
        results = predict_step([img])
        await ctx.message.reply(results[0])

bot.run(config['DISCORD_API_KEY'])
