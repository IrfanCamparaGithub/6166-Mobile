import io
import os
import cv2
import json
import time
import torch
import tempfile
import numpy as np

from pathlib import Path
from typing import Tuple, Optional

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse


ROOT = Path(".").resolve()
CACHE = Path(os.environ.get("XDG_CACHE_HOME", "/workspace/.cache"))
OUTPUT = Path(os.environ.get("OUTPUT_DIR", str(CACHE / "output")))
OUTPUT.mkdir(parents=True, exist_ok=True)


from app import load_landmarker, try_load_3d_stack  


def _create_video_writer(
    out_base: Path,
    w: int,
    h: int,
    fps: float,
    prefer: Optional[str] = None,
):
    
    prefer = (prefer or "").lower()

    webm = [
        ("VP90", out_base.with_suffix(".webm")),  
        ("VP80", out_base.with_suffix(".webm")),  
    ]
    mp4 = [
        ("mp4v", out_base.with_suffix(".mp4")),   
    ]
    avi = [("MJPG", out_base.with_suffix(".avi"))]

    candidates = (mp4 + webm + avi) if prefer == "mp4" else (webm + mp4 + avi)

    for fourcc_str, out_path in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(w), int(h)))
        if writer.isOpened():
            return writer, out_path
    return None, None


app = FastAPI(title="SMIRK API", version="0.3 (3D only, MP4-prefer)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

log: list[str] = []


detectors = load_landmarker(Path("assets/face_landmarker.task"), log)
detect_img = detectors["image"]
make_video_detector = detectors["make_video"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
stack = try_load_3d_stack(DEVICE, log)  

def _crop_square_from_landmarks(img_bgr: np.ndarray, lm: np.ndarray, scale: float = 1.25) -> np.ndarray:
    
    h, w = img_bgr.shape[:2]
    x0, y0 = int(lm[:, 0].min()), int(lm[:, 1].min())
    x1, y1 = int(lm[:, 0].max()), int(lm[:, 1].max())
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    size = int(max(x1 - x0, y1 - y0) * scale)
    x0, y0 = max(cx - size // 2, 0), max(cy - size // 2, 0)
    x1, y1 = min(cx + size // 2, w), min(cy + size // 2, h)
    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return img_bgr
    return crop

def _render_3d_rgb(crop_bgr: np.ndarray):
    
    if stack is None:
        return None
    smirk, flame, renderer = stack
    crop = cv2.resize(crop_bgr, (224, 224))
    ten = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None] / 255.0
    ten = ten.to(DEVICE)
    with torch.no_grad():
        out3 = smirk(ten)
        f = flame.forward(out3)
        ren = renderer.forward(
            f["vertices"], out3["cam"],
            landmarks_fan=f.get("landmarks_fan"),
            landmarks_mp=f.get("landmarks_mp"),
        )
        rendered = (ren["rendered_img"].clamp(0, 1) * 255).byte()[0].permute(1, 2, 0).cpu().numpy()
    return rendered  

def _letterbox_center(rgb: np.ndarray, W: int, H: int) -> np.ndarray:
    
    canvas = np.zeros((H, W, 3), np.uint8)
    h, w = rgb.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    rsz = cv2.resize(rgb, (nw, nh))
    x0 = (W - nw) // 2
    y0 = (H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = rsz
    return canvas


@app.get("/")
def root():
    return {"ok": True, "msg": "SMIRK API (3D-only) running", "try": ["/health", "POST /process/image", "POST /process/video", "/docs"]}

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "has_stack": bool(stack),
        "log_tail": log[-8:],
    }

@app.post("/process/image")
async def process_image(file: UploadFile = File(...)):
   
    if stack is None:
        return JSONResponse({"error": "3d_unavailable"}, status_code=503)

    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "decode_failed"}, status_code=400)

    lm = detect_img(img)
    if lm is None:
        return JSONResponse({"error": "no_face_detected"}, status_code=422)

    crop = _crop_square_from_landmarks(img, lm, scale=1.25)
    rgb = _render_3d_rgb(crop)
    if rgb is None:
        return JSONResponse({"error": "render_failed"}, status_code=500)

    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return JSONResponse({"error": "encode_failed"}, status_code=500)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")

@app.post("/process/video")
async def process_video(
    request: Request,
    file: UploadFile = File(...),
    stride: int = Form(2),
    format: Optional[str] = Form(None),
):
    
    if stack is None:
        return JSONResponse({"error": "3d_unavailable"}, status_code=503)

   
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        tmp.write(await file.read())
        src_path = Path(tmp.name)

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return JSONResponse({"error": "open_video_failed"}, status_code=400)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    
    ua = request.headers.get("user-agent", "").lower()
    prefer = "mp4" if (str(format).lower() == "mp4" or "iphone" in ua or "ipad" in ua or "ipod" in ua) else None

    writer, out_path = _create_video_writer(OUTPUT / f"{src_path.stem}_3d", W, H, fps, prefer=prefer)
    if writer is None:
        cap.release()
        return JSONResponse({"error": "writer_init_failed"}, status_code=500)

    detect_video = make_video_detector()
    ts_ms = 0
    step_ms = int(round(1000.0 / fps))
    fidx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if fidx % max(1, int(stride)) == 0:
            lm = detect_video(frame, ts_ms)
            if lm is not None:
                crop = _crop_square_from_landmarks(frame, lm, scale=1.25)
                rgb = _render_3d_rgb(crop)
                if rgb is not None:
                    out_bgr = cv2.cvtColor(_letterbox_center(rgb, W, H), cv2.COLOR_RGB2BGR)
                else:
                    out_bgr = np.zeros((H, W, 3), np.uint8)
            else:
                out_bgr = np.zeros((H, W, 3), np.uint8)  
        else:
            
            out_bgr = np.zeros((H, W, 3), np.uint8)

        writer.write(out_bgr)
        fidx += 1
        ts_ms += step_ms

    cap.release()
    writer.release()

    url = str(request.base_url).rstrip("/") + "/download/" + out_path.name
    return JSONResponse({"output_url": url, "filename": out_path.name})

@app.get("/download/{fname}")
def download(fname: str):
    path = OUTPUT / fname
    if not path.exists():
        return JSONResponse({"error": "not_found"}, status_code=404)
    mime = "video/webm" if path.suffix == ".webm" else ("video/mp4" if path.suffix == ".mp4" else "video/x-msvideo")
    resp = FileResponse(path, media_type=mime, filename=path.name)
    
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp
