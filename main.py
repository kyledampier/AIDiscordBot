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

testing = True

print(f"GPU is available: {torch.cuda.is_available()}")

config_path = "config.json"
if testing:
    config_path = "config.test.json"

print(f"Using config: {config_path}")
config = json.load(open(config_path))

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

# Stable Diffusion
print("Loading Stable Diffusion")
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=config['HUGGINGFACE_API_KEY'])
pipe = pipe.to(device)

# Comment out unneed code
'''
# GPT-2
print("Loading GPT-2")
generator = pipeline('text-generation', model="gpt2")
set_seed(42)

# Image Captioning
print("Loading GPT-2 image captioning")
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
'''

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

bot = commands.Bot(command_prefix='-', intents=intents)

@bot.event
async def on_ready():
    for guild in bot.guilds:
        for channel in guild.channels:
            if channel.name == "bots":
                await channel.send("@here **I am now online!**")
'''
@bot.command()
async def complete(ctx, *args):
    """Completes written prompt using GPT-2"""
    if ctx.channel.name != "bots":
        return

    if len(args) == 0:
        await ctx.send("Please follow the format `-complete <prompt>`")
        return

    prompt = " ".join(args)
    output = generator(prompt, max_length=120, num_return_sequences=1)
    await ctx.channel.send(f"Completed phrase for {ctx.author.mention}:\n{output[0]['generated_text']}")
'''

@bot.command()
async def generate(ctx, *args):
    """Generates two images based on your prompt using Stable Diffusion 1.4"""
    if ctx.channel.name != "bots":
        return

    if len(args) == 0:
        await ctx.send("Please follow the format `-generate <prompt>`")
        return

    prompt = " ".join(args)
    loading_msg = await ctx.channel.send(f"Generating **{prompt}** for {ctx.author.mention}... (0/1)")

    t = datetime.now().strftime("%Y%m%d-%H%M%S")
    files = []
    scales = [8.5]

    for i in range(len(scales)):
        with autocast("cuda"):
            image = pipe(prompt, guidance_scale=scales[i], num_inference_steps=50).images[0]

        file_path = os.path.join("images", f"{t}-{i}.png")
        image.save(file_path)
        files.append(discord.File(file_path))
        # await loading_msg.edit(content=f"Generating images for {ctx.author.mention}... ({i+1}/{len(scales)})")

    await ctx.channel.send(f"**{prompt}** (requested by {ctx.author.mention})", files=files)
    await loading_msg.delete()

@bot.command()
async def change(ctx, *args):
    """Chnages the attached image using Stable Diffusion"""
    if ctx.channel.name != "bots":
        return

    if len(args) > 0:
        prompt = " ".join(args)
    else:
        await ctx.send("No prompt found. Please follow the format `-change <prompt> <attachment>`")
        return

    if len(ctx.message.attachments) > 0:
        attached_file = ctx.message.attachments[0]
        url = attached_file
    else:
        await ctx.send("No image attachment found. Please follow the format `-change <prompt> <attachment>`")
        return

    try:
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content))
        img = img.convert('RGB')
        img = img.resize((512, 512))
    except:
        await ctx.send("Error: Retrieving image failed")
    else:
        t = datetime.now().strftime("%Y%m%d-%H%M%S")
        with autocast("cuda"):
            image = pipe(prompt, guidance_scale=7.0, num_inference_steps=25, init_image=img, strength=0.50).images[0]

        file_path = os.path.join("images", f"{t}-sourced.png")
        image.save(file_path)
        file = discord.File(file_path)
        await ctx.message.reply(files=[file])

'''
@bot.command()
async def describe(ctx, *args):
    """Describes the message's attachment or image_url into words"""
    if ctx.channel.name != "bots":
        return

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
        img = img.convert('RGB')
    except:
        await ctx.send("Error: Retrieving image from URL failed")
    else:
        results = predict_step([img])
        await ctx.message.reply(results[0])
'''

@bot.command()
@commands.is_owner()
async def shutdown(ctx):
    await ctx.channel.send("@here **Shutting down!**")
    await bot.close()

bot.run(config['DISCORD_API_KEY'])
