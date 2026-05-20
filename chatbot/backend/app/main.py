from __future__ import annotations

import json
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.agent import agent
from app.config import backend_config

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NPY_EXTENSIONS = {".npy"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | NPY_EXTENSIONS


def _image_to_npy(img_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(img_bytes)).convert("L")
    img = img.resize((240, 240), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.stack([arr] * 4, axis=0)

app = FastAPI(title="BraTS Tumor Analysis Chatbot", version="1.0.0")

cors = backend_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors.get("cors_origins", ["http://localhost:5173"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FileInfo(BaseModel):
    file_path: str
    filename: str
    shape: list[int] = []


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    files: list[FileInfo] = []
    session_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    msgs = [m.model_dump() for m in req.messages]
    if req.files:
        lines = []
        for f in req.files:
            shape_str = f"[{', '.join(str(s) for s in f.shape)}]" if f.shape else ""
            lines.append(f"- {f.filename} {shape_str} at {f.file_path}")
        context = f"Uploaded files for analysis:\n" + "\n".join(lines)
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1]["content"] = f"{context}\n\n{msgs[-1]['content']}" if msgs[-1]["content"] else context
        else:
            msgs.append({"role": "user", "content": context})

    result = agent.invoke(msgs, session_id=req.session_id)
    return result


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    msgs = [m.model_dump() for m in req.messages]
    if req.files:
        lines = []
        for f in req.files:
            shape_str = f"[{', '.join(str(s) for s in f.shape)}]" if f.shape else ""
            lines.append(f"- {f.filename} {shape_str} at {f.file_path}")
        context = f"Uploaded files for analysis:\n" + "\n".join(lines)
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1]["content"] = f"{context}\n\n{msgs[-1]['content']}" if msgs[-1]["content"] else context
        else:
            msgs.append({"role": "user", "content": context})

    def event_generator():
        for event in agent.stream(msgs, session_id=req.session_id):
            yield {"event": event["type"], "data": json.dumps(event)}

    return EventSourceResponse(event_generator())


@app.post("/upload")
async def upload_file(file: UploadFile):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {"error": f"Unsupported file type '{ext}'. Upload .npy, .jpg, .jpeg, or .png."}

    session_dir = UPLOAD_DIR / uuid.uuid4().hex[:8]
    session_dir.mkdir(parents=True, exist_ok=True)
    content = await file.read()

    if ext in NPY_EXTENSIONS:
        dest = session_dir / file.filename
        dest.write_bytes(content)
        arr = np.load(dest)
    else:
        stem = Path(file.filename).stem
        dest = session_dir / f"{stem}.npy"
        arr = _image_to_npy(content)
        np.save(dest, arr)

    info = {
        "file_path": str(dest.resolve()),
        "filename": f"{Path(file.filename).stem}.npy",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size_kb": round(len(content) / 1024, 1),
    }

    return info


@app.post("/session")
def new_session():
    return {"session_id": str(uuid.uuid4())}
