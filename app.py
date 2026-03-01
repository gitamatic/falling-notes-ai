from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline
import torch
import base64
from io import BytesIO

app = FastAPI()

class Request(BaseModel):
    prompt: str

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.to("cuda")
pipe.enable_model_cpu_offload()

@app.post("/generate")
async def generate(data: Request):
    image = pipe(
        data.prompt,
        num_inference_steps=25,
        height=1024,
        width=1024
    ).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return {
        "image": base64.b64encode(buffer.getvalue()).decode()
    }
