import io
from fastapi import FastAPI, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse

# FastAPI is effectively Starlette + Pydantic
from pydantic import BaseModel

from ml import obtain_image

app = FastAPI()

app.mount("/static", StaticFiles(directory="./static"), name="static")


@app.get("/")
def index():
    with open("templates/index.html") as f:
        html = f.read()
    return HTMLResponse(html)


# @app.get("/items/{item_id}")
# def read_item(item_id: int):
#     item_id
#     return {"item_id": item_id}


# class Item(BaseModel):
#     name: str
#     price: float
#     tags: list[str] = []


# @app.post("/items/")
# def create_item(item: Item):
#     return item
class ImageRequest(BaseModel):
    prompt_text: str


@app.post("/generate-image")
async def generate_image(
    prompt_text: ImageRequest,
    seed: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
):

    prompt = prompt_text.dict()["prompt_text"]
    print(prompt)
    image = obtain_image(
        prompt,
        num_inference_steps=num_inference_steps,
        seed=seed,
        guidance_scale=guidance_scale,
    )
    image.save("static/image.jpg")
    return File("static/image.jpg", content_type="image/jpeg")


@app.get("/generate")
def generate_image_memory(
    prompt: str,
    *,
    seed: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
):
    image = obtain_image(
        prompt,
        num_inference_steps=num_inference_steps,
        seed=seed,
        guidance_scale=guidance_scale,
    )
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")


@app.get("/generate/{prompt}")
def generate_image_memory(
    prompt: str,
    *,
    seed: int | None = None,
    num_steps: int = 50,
    guidance_scale: float = 7.5
):
    prompt = prompt.replace("_", " ")

    image = obtain_image(
        prompt,
        num_inference_steps=num_steps,
        seed=seed,
        guidance_scale=guidance_scale,
    )
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")
