"""Small local Diffusers service with an AUTOMATIC1111-compatible endpoint."""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from firephantom.runtime import memory_status

_lock = threading.Lock()
_pipeline = None
_img2img = None
_device = "unloaded"
NEGATIVE_PROMPT = (
    "blurry, out of focus, soft focus, abstract, gradient, color fog, "
    "smooth blob, low resolution, watermark, text, logo, cartoon"
)


class Txt2ImgRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(NEGATIVE_PROMPT, max_length=1000)
    steps: int = Field(20, ge=1, le=20)
    width: int = Field(256, ge=64, le=512)
    height: int = Field(256, ge=64, le=512)
    cfg_scale: float = Field(7.0, ge=0.0, le=20.0)
    seed: int = Field(0, ge=0, le=2**32 - 1)


class Img2ImgRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = Field(NEGATIVE_PROMPT, max_length=1000)
    steps: int = Field(20, ge=1, le=20)
    width: int = Field(256, ge=64, le=512)
    height: int = Field(256, ge=64, le=512)
    cfg_scale: float = Field(7.0, ge=0.0, le=20.0)
    seed: int = Field(0, ge=0, le=2**32 - 1)
    strength: float = Field(0.2, ge=0.0, le=1.0)
    init_images: list[str] = Field(..., min_length=1)
    # White pixels may change. Tiny-SD is not an inpainting model, so the
    # result is blended back over the init image through this mask.
    mask: str | None = None
    mask_blur: int = Field(4, ge=0, le=64)


def _device_and_dtype(torch):
    if torch.backends.mps.is_available():
        # Tiny-SD produces all-zero pixels with float16 on Apple MPS.
        return "mps", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _check_memory_headroom() -> None:
    """Refuse a cold model load when macOS reports too little headroom."""
    if os.environ.get("FIREPHANTOM_MEMORY_GUARD", "1").lower() in {"0", "false", "no"}:
        return
    status = memory_status()
    if status["state"] == "critical":
        available = status.get("available_gib")
        minimum = status.get("minimum_available_gib")
        raise RuntimeError(
            f"Stable Diffusion load blocked by memory guard: "
            f"available {available if available is not None else 'unknown'} GiB, "
            f"minimum {minimum:g} GiB; stop other MPS jobs first"
        )


def _load_pipeline():
    global _pipeline, _device
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline
        try:
            import torch
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise RuntimeError("Install requirements-ml.txt to enable local Stable Diffusion") from exc
        model_path = Path(os.environ.get("SD_MODEL_PATH", "/Volumes/T7/FirePhantom-runtime/models/tiny-sd"))
        if not model_path.exists():
            raise RuntimeError(f"Stable Diffusion model directory does not exist: {model_path}")
        _check_memory_headroom()
        _device, dtype = _device_and_dtype(torch)
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype, safety_checker=None)
        pipe.to(_device)
        pipe.set_progress_bar_config(disable=True)
        if _device != "cpu":
            pipe.enable_attention_slicing()
        _pipeline = pipe
        return _pipeline


def _build_img2img(txt_pipe):
    """Wrap the already loaded Tiny-SD weights. Do not load a second copy."""
    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline(
        vae=txt_pipe.vae,
        text_encoder=txt_pipe.text_encoder,
        tokenizer=txt_pipe.tokenizer,
        unet=txt_pipe.unet,
        scheduler=txt_pipe.scheduler,
        safety_checker=None,
        feature_extractor=getattr(txt_pipe, "feature_extractor", None),
        image_encoder=getattr(txt_pipe, "image_encoder", None),
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _img2img_pipeline():
    global _img2img
    txt_pipe = _load_pipeline()
    if _img2img is not None:
        return _img2img
    with _lock:
        if _img2img is not None:
            return _img2img
        _img2img = _build_img2img(txt_pipe)
        return _img2img


def _decode_init(encoded: str, mode: str = "RGB"):
    from PIL import Image

    if "," in encoded[:64]:
        encoded = encoded.split(",", 1)[1]
    raw = base64.b64decode(encoded)
    return Image.open(BytesIO(raw)).convert(mode)


def _blend_masked(init, generated, encoded_mask: str | None, blur: int):
    """Keep init pixels where the mask is black. The flame lives there."""
    if not encoded_mask:
        return generated
    from PIL import Image, ImageFilter

    mask = _decode_init(encoded_mask, "L").resize(init.size)
    if blur:
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
    return Image.composite(generated.resize(init.size), init, mask)


@asynccontextmanager
async def lifespan(_application: FastAPI):
    yield


app = FastAPI(title="FirePhantom Local Stable Diffusion", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "loaded": _pipeline is not None, "device": _device}


@app.post("/sdapi/v1/txt2img")
def txt2img(request: Txt2ImgRequest) -> dict:
    try:
        import torch
        pipe = _load_pipeline()
        generator = torch.Generator(device="cpu").manual_seed(request.seed)
        result = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.cfg_scale,
            width=request.width - request.width % 8,
            height=request.height - request.height % 8,
            generator=generator,
        )
        output = BytesIO()
        result.images[0].save(output, format="PNG")
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return {"images": [encoded], "device": _device, "seed": request.seed}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/sdapi/v1/img2img")
def img2img(request: Img2ImgRequest) -> dict:
    try:
        import torch
        pipe = _img2img_pipeline()
        width = request.width - request.width % 8
        height = request.height - request.height % 8
        image = _decode_init(request.init_images[0]).resize((width, height))
        generator = torch.Generator(device="cpu").manual_seed(request.seed)
        result = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            image=image,
            strength=request.strength,
            num_inference_steps=request.steps,
            guidance_scale=request.cfg_scale,
            generator=generator,
        )
        blended = _blend_masked(image, result.images[0], request.mask, request.mask_blur)
        output = BytesIO()
        blended.save(output, format="PNG")
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return {"images": [encoded], "device": _device, "seed": request.seed}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


def main() -> None:
    import uvicorn
    uvicorn.run("firephantom.synthesis.diffusion_server:app", host=os.environ.get("SD_HOST", "127.0.0.1"), port=int(os.environ.get("SD_PORT", "7861")))


if __name__ == "__main__":
    main()
