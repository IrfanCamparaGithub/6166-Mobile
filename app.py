
import os
from pathlib import Path
import importlib
import numpy as np
import cv2


import torch
import streamlit as st
import mediapipe as mp
from itertools import chain


CACHE_DIR = Path("/workspace/.cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("CUDA_CACHE_PATH", str(CACHE_DIR))
Path(".cache").mkdir(exist_ok=True)

ASSETS  = Path("assets");  ASSETS.mkdir(exist_ok=True)
SAMPLES = Path("samples")
OUTPUT  = Path(os.environ.get("OUTPUT_DIR", "/workspace/.cache/output"))

OUTPUT.mkdir(parents=True, exist_ok=True)

TASK_PATH = ASSETS / "face_landmarker.task"

















CKPT_CANDIDATES = [Path("pretrained_models/SMIRK_em1.pt"), Path("trained_models/SMIRK_em1.pt")]
FLAME_PKL = ASSETS / "FLAME2020/generic_model.pkl"
HEAD_OBJ  = ASSETS / "head_template.obj"

# ---------- video writer (prefer H.264 to avoid VP9 warnings) ----------
def _create_video_writer(out_base: Path, w: int, h: int, fps: float):
    """
    Returns (writer, out_path). Browser-friendly preference:
      1) MP4 (H.264 via avc1/H264/X264)
      2) MP4 (mp4v)
      3) WebM (VP9/VP8)
      4) AVI (MJPG)
    """
    candidates = [
        ("avc1", out_base.with_suffix(".mp4")),
        ("H264", out_base.with_suffix(".mp4")),
        ("X264", out_base.with_suffix(".mp4")),
        ("mp4v", out_base.with_suffix(".mp4")),
        ("VP90", out_base.with_suffix(".webm")),
        ("VP80", out_base.with_suffix(".webm")),
        ("MJPG", out_base.with_suffix(".avi")),
    ]




    for fourcc_str, out_path in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, out_path
    return None, None

# ---------- quick runtime diagnostics ----------
def _p3d_diag():
    info = []
    info.append(f"torch: {torch.__version__}, cuda: {torch.version.cuda}, is_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            info.append(f"device: {torch.cuda.get_device_name(0)}  cc: {torch.cuda.get_device_capability(0)}")
        except Exception as e:
            info.append(f"device query failed: {e}")
    try:
        _C = importlib.import_module("pytorch3d._C")
        has_cuda_ops = any("cuda" in k.lower() for k in dir(_C))
        syms = [k for k in dir(_C) if any(n in k for n in ("rasterize_meshes", "cubify"))]
        info.append(f"pytorch3d._C import: OK  (cuda ops present: {has_cuda_ops})  subset: {syms or 'none'}")
    except Exception as e:
        info.append(f"pytorch3d._C import failed: {e!r}")
    return "\n".join(info)

# ---------- mesh drawing ----------
MESH_CONNECTIONS = list(chain(
    map(tuple, mp.solutions.face_mesh.FACEMESH_TESSELATION),
    map(tuple, mp.solutions.face_mesh.FACEMESH_CONTOURS),
    map(tuple, mp.solutions.face_mesh.FACEMESH_IRISES),
))
def draw_mesh(img_bgr, lm, color=(0,255,0), th=1):
    out = img_bgr.copy()
    pts = lm.astype(int)
    for a,b in MESH_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(out, (pts[a,0],pts[a,1]), (pts[b,0],pts[b,1]), color, th, cv2.LINE_AA)
    return out

# ---------- detectors ----------
def load_landmarker(task_path: Path, log):



















































    """
    Returns:
      {
        "image": callable(image_bgr) -> (Nx3) or None,
        "make_video": callable() -> callable(image_bgr, timestamp_ms:int) -> (Nx3) or None
      }
    """
    if task_path.exists():
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            base = python.BaseOptions(model_asset_path=str(task_path))

            img_opts = vision.FaceLandmarkerOptions(
                base_options=base, num_faces=1,
                output_face_blendshapes=True,
                running_mode=vision.RunningMode.IMAGE
            )
            vid_opts = vision.FaceLandmarkerOptions(
                base_options=base, num_faces=1,
                output_face_blendshapes=True,
                running_mode=vision.RunningMode.VIDEO
            )

            lm_img = vision.FaceLandmarker.create_from_options(img_opts)
            log.append(f"✓ Using cached task at {task_path}")

            def _image(image_bgr):
                rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = lm_img.detect(mp_img)
                if not res.face_landmarks: return None
                h, w = rgb.shape[:2]; L = res.face_landmarks[0]
                arr = np.zeros((len(L),3), np.float32)
                for i,p in enumerate(L): arr[i] = [p.x*w, p.y*h, p.z]
                return arr

            def make_video():
                lm_vid = vision.FaceLandmarker.create_from_options(vid_opts)
                def _video(image_bgr, timestamp_ms:int):
                    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    res = lm_vid.detect_for_video(mp_img, timestamp_ms=timestamp_ms)
                    if not res.face_landmarks: return None
                    h, w = rgb.shape[:2]; L = res.face_landmarks[0]
                    arr = np.zeros((len(L),3), np.float32)
                    for i,p in enumerate(L): arr[i] = [p.x*w, p.y*h, p.z]
                    return arr
                return _video

            return {"image": _image, "make_video": make_video}

        except Exception as e:
            log.append(f"⚠ Tasks API init failed, using FaceMesh fallbacks: {e}")

    # Fallbacks
    fm_image = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,  max_num_faces=1, refine_landmarks=True)
    log.append("ℹ Using FaceMesh fallback (image + video)")

    def _image(image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = fm_image.process(rgb)
        if not res.multi_face_landmarks: return None
        h, w = rgb.shape[:2]; L = res.multi_face_landmarks[0].landmark
        arr = np.zeros((len(L),3), np.float32)
        for i,p in enumerate(L): arr[i] = [p.x*w, p.y*h, p.z]
        return arr

    def make_video():
        fm_video = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        def _video(image_bgr, timestamp_ms:int):  # timestamp unused here
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            res = fm_video.process(rgb)
            if not res.multi_face_landmarks: return None
            h, w = rgb.shape[:2]; L = res.multi_face_landmarks[0].landmark
            arr = np.zeros((len(L),3), np.float32)
            for i,p in enumerate(L): arr[i] = [p.x*w, p.y*h, p.z]
            return arr
        return _video

    return {"image": _image, "make_video": make_video}

def try_load_3d_stack(device: str, log):
    ckpt = next((p for p in CKPT_CANDIDATES if p.exists()), None)
    if not ckpt:
        log.append("ℹ Missing SMIRK checkpoint in pretrained_models/")
        return None
    if not FLAME_PKL.exists():
        log.append("ℹ Missing FLAME: assets/FLAME2020/generic_model.pkl")
        return None
    if not HEAD_OBJ.exists():
        log.append("ℹ Missing head_template.obj in assets/")
        return None
    if device != "cuda":
        log.append("ℹ CUDA not available, 3D disabled")
        return None
    try:
        import pytorch3d  # noqa: F401
    except Exception as e:
        log.append(f"ℹ PyTorch3D not available: {e}")
        return None

    import sys; sys.path.append(str(Path('.').resolve()))
    try:
        from src.smirk_encoder import SmirkEncoder
        from src.FLAME.FLAME import FLAME
        from src.renderer.renderer import Renderer
    except Exception as e:
        log.append(f"⚠ Could not import SMIRK/FLAME modules: {e}")
        return None

    try:
        smirk = SmirkEncoder().to(device).eval()
        data  = torch.load(ckpt, map_location=device)
        state = data.get("state_dict", data)
        if any(k.startswith("smirk_encoder.") for k in state):
            state = {k.replace("smirk_encoder.",""): v for k,v in state.items()}
        smirk.load_state_dict(state, strict=False)

        flame = FLAME(flame_model_path=str(FLAME_PKL)).to(device).eval()
        renderer = Renderer(render_full_head=False, obj_filename=str(HEAD_OBJ)).to(device).eval()
        return smirk, flame, renderer
    except Exception as e:
        log.append(f"⚠ Could not initialize 3D stack: {e}")
        return None

# ================================
# Streamlit UI (only when run directly)
# ================================
if __name__ == "__main__":
    st.set_page_config(page_title="SMIRK — Auto Face Landmark + 3D", layout="wide")
    st.title("🧠 SMIRK — Auto Face Landmark + 3D")
    st.sidebar.header("Runtime diagnostics")
    st.sidebar.code(_p3d_diag())

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Device: {DEVICE.upper()}")

    log = []
    detectors   = load_landmarker(TASK_PATH, log)
    detect_img  = detectors["image"]
    make_video  = detectors["make_video"]
    stack = try_load_3d_stack(DEVICE, log)
    if stack: st.success("3D stack ready.")
    else:     st.warning("3D disabled — see Log below for what’s missing.")

    st.sidebar.markdown("### Video options")
    FRAME_STRIDE = st.sidebar.number_input("Frame stride (process every Nth frame)", min_value=1, max_value=10, value=2, step=1)
    VIDEO_3D     = st.sidebar.checkbox("Render 3D on videos (slower)", value=bool(stack))
    SHOW_PLAYERS = st.sidebar.checkbox("Show inline video players", value=True)

    # ---------- images ----------
    imgs = sorted(list(SAMPLES.glob("*.png")) + list(SAMPLES.glob("*.jpg")) + list(SAMPLES.glob("*.jpeg")))
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None: continue

        lm = detect_img(img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"{p.name} — original")

        rhs_img = None
        if lm is not None:
            mesh = draw_mesh(img, lm)
            rhs_img = cv2.cvtColor(mesh, cv2.COLOR_BGR2RGB)

        if stack is not None and lm is not None:
            try:
                smirk, flame, renderer = stack
                x0, y0 = int(lm[:,0].min()), int(lm[:,1].min())
                x1, y1 = int(lm[:,0].max()), int(lm[:,1].max())
                cx, cy = (x0+x1)//2, (y0+y1)//2
                size = int(max(x1-x0, y1-y0) * 1.25)
                x0, y0 = max(cx-size//2, 0), max(cy-size//2, 0)
                x1, y1 = min(cx+size//2, img.shape[1]), min(cy+size//2, img.shape[0])
                crop = img[y0:y1, x0:x1]
                crop = cv2.resize(crop, (224,224))
                ten = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).permute(2,0,1).float()[None]/255.0
                ten = ten.to(DEVICE)
                with torch.no_grad():
                    out = smirk(ten)
                    f = flame.forward(out)
                    ren = renderer.forward(
                        f["vertices"], out["cam"],
                        landmarks_fan=f.get("landmarks_fan"),
                        landmarks_mp=f.get("landmarks_mp"),
                    )
                    rendered = (ren["rendered_img"].clamp(0,1)*255).byte()[0].permute(1,2,0).cpu().numpy()
                rhs_img = np.concatenate([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), rendered], axis=1)
            except Exception as e:
                log.append(f"⚠ 3D render failed on {p.name}: {e}")

        with col2:
            if rhs_img is not None:
                st.image(rhs_img, caption=f"{p.name} — {'3D render' if stack else '2D mesh'}")
                out_path = OUTPUT / p.name
                cv2.imwrite(str(out_path), cv2.cvtColor(rhs_img, cv2.COLOR_RGB2BGR))
                st.write(f"✅ Saved: {out_path}")
            else:
                st.warning(f"No face detected in {p.name}")

    # ---------- videos ----------
    video_exts = ("*.mp4","*.mov","*.avi","*.mkv","*.webm")
    videos = sorted(list(chain.from_iterable(SAMPLES.glob(e) for e in video_exts)))

    def process_video(path: Path, detect_video, stride:int=2, do_3d:bool=False):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            log.append(f"⚠ Could not open video: {path.name}")
            return None

        fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer, out_path = _create_video_writer(OUTPUT / f"{path.stem}_mesh", w, h, fps)
        if writer is None:
            log.append("⚠ Could not initialize any video writer.")
            cap.release()
            return None

        preview_path = out_path.with_suffix(".preview.jpg")
        wrote_preview = False

        smirk = flame = renderer = None
        if do_3d and stack is not None:
            smirk, flame, renderer = stack

        frame_idx = 0
        ts_ms = 0
        step_ms = int(round(1000.0 / fps))

        while True:
            ok, frame = cap.read()
            if not ok: break

            draw = frame
            if frame_idx % stride == 0:
                lm = detect_video(frame, ts_ms)
                if lm is not None:
                    draw = draw_mesh(frame, lm)
                    if do_3d and smirk is not None:
                        try:
                            x0, y0 = int(lm[:,0].min()), int(lm[:,1].min())
                            x1, y1 = int(lm[:,0].max()), int(lm[:,1].max())
                            cx, cy = (x0+x1)//2, (y0+y1)//2
                            size = int(max(x1-x0, y1-y0) * 1.25)
                            x0, y0 = max(cx-size//2, 0), max(cy-size//2, 0)
                            x1, y1 = min(cx+size//2, frame.shape[1]), min(cy+size//2, frame.shape[0])
                            crop = frame[y0:y1, x0:x1]
                            crop = cv2.resize(crop, (224,224))
                            ten = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).permute(2,0,1).float()[None]/255.0
                            ten = ten.to(DEVICE)
                            with torch.no_grad():
                                out = smirk(ten)
                                f = flame.forward(out)
                                ren = renderer.forward(
                                    f["vertices"], out["cam"],
                                    landmarks_fan=f.get("landmarks_fan"),
                                    landmarks_mp=f.get("landmarks_mp"),
                                )
                                rendered = (ren["rendered_img"].clamp(0,1)*255).byte()[0].permute(1,2,0).cpu().numpy()
                            rsz = cv2.resize(rendered, (min(256, w//3), min(256, h//3)))
                            draw[10:10+rsz.shape[0], 10:10+rsz.shape[1]] = cv2.cvtColor(rsz, cv2.COLOR_RGB2BGR)
                        except Exception as e:
                            log.append(f"⚠ 3D render failed on frame {frame_idx} of {path.name}: {e}")

            if not wrote_preview:
                cv2.imwrite(str(preview_path), draw)
                wrote_preview = True

            writer.write(draw)
            frame_idx += 1
            ts_ms += step_ms



        cap.release()
        writer.release()
        return out_path

    if videos:
        st.header("🎬 Videos")
    for p in videos:
        st.write(f"Processing **{p.name}** …")
        detect_video = make_video()  # fresh detector per file
        try:
            outp = process_video(p, detect_video=detect_video, stride=FRAME_STRIDE, do_3d=VIDEO_3D)
            if outp is not None:
                st.success(f"Saved: {outp}")
                prev_img = outp.with_suffix(".preview.jpg")
                if SHOW_PLAYERS:
                    st.video(str(outp))
                elif prev_img.exists():
                    st.image(str(prev_img), caption=f"{p.name} — preview")
                mime = "video/webm" if outp.suffix.lower()==".webm" else ("video/mp4" if outp.suffix.lower()==".mp4" else "video/x-msvideo")
                with open(outp, "rb") as f:
                    st.download_button("⬇️ Download processed video", f, file_name=outp.name, mime=mime)

            else:
                st.warning(f"Skipped {p.name}")
        except Exception as e:
            st.exception(e)























    if log:
        st.subheader("Log")
        st.text("\n".join(log))