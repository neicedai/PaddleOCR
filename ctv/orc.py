#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 优先的本地 OCR HTTP 服务（PaddleOCR 3.x）
- 优先使用 GPU（device="gpu"），若 cuDNN/驱动不可用会自动回退到 CPU（可用 OCR_GPU_STRICT=1 禁止回退）
- 预热：启动时做一次小图推理以加载模型，避免首个请求卡顿
- 路由：
    POST /ocr      { "image_b64"|"file_b64": "<base64或dataURL>", "fileType"?: "pdf"|"image" }
                  -> { ok, lines:[{text, score, box, page}], error? }
    GET  /healthz  -> { ok: true }
    GET  /version  -> { paddleocr, paddlepaddle, device, use_gpu, cls, pdf_support, pdf_page_limit }
- 依赖：fastapi uvicorn pillow numpy pdf2image paddleocr（以及已装好的 paddlepaddle-gpu 3.x + cuDNN8、poppler）
建议：
    pip install -U fastapi uvicorn pillow numpy pdf2image
    pip install paddleocr>=3.0.0
    pip install "paddlepaddle-gpu==2.6.1" -f https://www.paddlepaddle.org.cn/whl/cu121.html
    # pdf2image 依赖系统级 poppler，可通过 apt/yum/brew 安装 poppler-utils
"""

import base64, io, os, sys, time, json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field, root_validator
from PIL import Image, ImageDraw
from pdf2image import convert_from_bytes
import uvicorn

from paddleocr import PaddleOCR
import pkg_resources
# ---- cuDNN 预加载 & 启动自检（务必放在 import PaddleOCR / Paddle 之前） ----
import os, sys, ctypes

# 让进程一定能看到系统的 cuDNN 8
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.getenv("LD_LIBRARY_PATH", "")
# 给 Paddle 一个明确的 cuDNN 目录（可选但很稳）
os.environ.setdefault("FLAGS_cudnn_dir", "/usr/lib/x86_64-linux-gnu")

# 预加载关键 so，避免后续延迟加载找不到
for so in ("libcudnn.so.8", "libcudnn_ops_infer.so.8", "libcudnn_cnn_infer.so.8"):
    try:
        ctypes.CDLL(so)
        print(f"[PRELOAD] OK {so}", file=sys.stderr)
    except OSError as e:
        print(f"[PRELOAD][WARN] {so} -> {e}", file=sys.stderr)

# 到这里再 import paddle 并做硬自检
import paddle
try:
    print(f"[CHECK] Paddle {paddle.__version__}, compiled_with_cuda={paddle.is_compiled_with_cuda()}", file=sys.stderr)
    print(f"[CHECK] CUDA={paddle.version.cuda()}, cuDNN={paddle.version.cudnn()}", file=sys.stderr)
except Exception as e:
    print("[CHECK][FATAL] cuDNN runtime not available:", e, file=sys.stderr)
    raise
# --------------------------------------------------------------------------

# ================== 环境参数 ==================
# 是否严格要求用 GPU（1=严格，不可回退；0/缺省=可回退）
GPU_STRICT   = os.getenv("OCR_GPU_STRICT", "0") == "1"
# 是否启用文本行方向分类，图像基本正向可关（0）以提速
USE_CLS      = os.getenv("OCR_CLS", "0") == "1"     # 默认关，加速
# 端口
PORT         = int(os.getenv("OCR_PORT", "8001"))
# 限制检测侧边（缩放上限），默认 960；小一些可提速
DET_MAX_SIDE = int(os.getenv("OCR_DET_LIMIT_SIDE", "960"))
# 识别批大小，CPU/GPU 可调 16~64 观察吞吐
REC_BATCH    = int(os.getenv("OCR_REC_BATCH", "32"))
# 强制设备（仅用于日志提示）：gpu / cpu（实际设备由 use_gpu & 环境决定）
PREF_DEV     = os.getenv("OCR_DEVICE", "gpu").lower()

# ================== 初始化与回退 ==================
def init_ocr_prefer_gpu() -> (PaddleOCR, bool):
    """优先 GPU 初始化；失败时（如 cuDNN 缺失）自动回退到 CPU，除非 GPU_STRICT=1"""
    # 优先尝试 GPU
    if PREF_DEV.startswith("gpu") or PREF_DEV == "auto":
        try:
            gpu_device = "gpu" if PREF_DEV == "auto" else PREF_DEV
            print(f"[INIT] Try GPU (device={gpu_device})...", file=sys.stderr, flush=True)
            ocr = PaddleOCR(
                lang="ch",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=USE_CLS,
                text_det_limit_side_len=DET_MAX_SIDE,
                text_recognition_batch_size=REC_BATCH,
                device=gpu_device,
            )
            # 做一次 tiny 预热确认可用（可捕获 cuDNN/驱动异常）
            _warmup_once(ocr, USE_CLS)
            print("[INIT] GPU OK.", file=sys.stderr, flush=True)
            return ocr, True
        except Exception as e:
            msg = str(e)
            print(f"[INIT][WARN] GPU init failed: {msg}", file=sys.stderr, flush=True)
            if GPU_STRICT:
                # 严格模式：直接抛错
                raise

    # 回退 CPU
    print("[INIT] Fallback to CPU (device=cpu)...", file=sys.stderr, flush=True)
    ocr = PaddleOCR(
        lang="ch",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=USE_CLS,
        text_det_limit_side_len=DET_MAX_SIDE,
        text_recognition_batch_size=max(16, min(REC_BATCH, 64)),
        device="cpu",
    )
    _warmup_once(ocr, USE_CLS)
    print("[INIT] CPU OK.", file=sys.stderr, flush=True)
    return ocr, False

def _warmup_once(ocr: PaddleOCR, use_cls: bool):
    """用一张小图预热，触发模型加载，避免首个请求卡顿"""
    try:
        img = Image.new("RGB", (320, 160), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((10, 60), "你好，OCR", fill=(0, 0, 0))
        arr = np.array(img)
        t0 = time.time()
        _ = ocr.predict(arr, use_textline_orientation=use_cls)
        dt = time.time() - t0
        print(f"[WARMUP] done in {dt:.2f}s (cls={use_cls})", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[WARMUP][WARN] {e}", file=sys.stderr, flush=True)

ocr, USING_GPU = init_ocr_prefer_gpu()

# ================== FastAPI 层 ==================
app = FastAPI(title="Local OCR Service (GPU-first, PaddleOCR 3.x)")

class OcrReq(BaseModel):
    image_b64: Optional[str] = None  # 兼容旧字段，支持纯 base64 或 dataURL（data:image/...;base64,xxxx）
    file_b64: Optional[str] = None   # 新字段，便于传除图片以外的文件（如 PDF）
    fileType: Optional[str] = None   # 可选明确文件类型，pdf/image/...

    @root_validator
    def _ensure_payload(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not (values.get("file_b64") or values.get("image_b64")):
            raise ValueError("image_b64 或 file_b64 必须提供一个")
        return values

class OcrLine(BaseModel):
    text: str
    score: float
    box: Optional[List[List[float]]] = None
    page: Optional[int] = None

class OcrPage(BaseModel):
    page: int
    lines: List[OcrLine] = Field(default_factory=list)
    text: str = ""


class OcrResp(BaseModel):
    ok: bool
    lines: List[OcrLine] = Field(default_factory=list)
    pages: List[OcrPage] = Field(default_factory=list)
    error: Optional[str] = None

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/version")
def version():
    try:
        return {
            "paddleocr": pkg_resources.get_distribution("paddleocr").version,
            "paddlepaddle": pkg_resources.get_distribution("paddlepaddle-gpu").version
                              if USING_GPU else pkg_resources.get_distribution("paddlepaddle").version,
            "device": "gpu" if USING_GPU else "cpu",
            "use_gpu": USING_GPU,
            "cls": USE_CLS,
            "det_limit_side_len": DET_MAX_SIDE,
            "rec_batch_num": REC_BATCH,
            "pdf_support": True,
            "pdf_page_limit": None,
        }
    except Exception:
        return {
            "paddleocr": "unknown",
            "paddlepaddle": "unknown",
            "device": "gpu" if USING_GPU else "cpu",
            "use_gpu": USING_GPU,
            "cls": USE_CLS,
            "pdf_support": True,
            "pdf_page_limit": None,
        }

def _split_data_url(b64: str) -> Tuple[str, Optional[str]]:
    """将 dataURL 拆分为 base64 主体与 mime 信息"""
    data = b64.strip()
    mime = None
    if data.lower().startswith("data:"):
        header, _, payload = data.partition(",")
        if not payload:
            raise ValueError("dataURL 缺少数据部分")
        header = header[5:]  # 去掉 data:
        if ";" in header:
            mime = header.split(";", 1)[0].strip().lower()
        else:
            mime = header.strip().lower()
        data = payload
    return data, mime


def _decode_base64_data(b64: str) -> Tuple[bytes, Optional[str]]:
    """解码 base64（或 dataURL）内容并返回原始字节与可能的 mime"""
    payload, mime = _split_data_url(b64)
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise ValueError(f"无法解码 base64 数据: {exc}")
    return raw, mime


def _is_pdf_b64(b64: str, *, decoded_bytes: Optional[bytes] = None, mime_hint: Optional[str] = None) -> bool:
    """判断 base64 内容是否为 PDF。若提供 decoded_bytes/mime_hint 可避免重复解码"""
    raw = decoded_bytes
    mime = mime_hint
    if raw is None:
        try:
            raw, mime = _decode_base64_data(b64)
        except Exception:
            return False
    if mime and mime.lower() == "application/pdf":
        return True
    if raw is None:
        return False
    return raw.lstrip().startswith(b"%PDF")


def _read_image_from_b64(b64: str, *, decoded_bytes: Optional[bytes] = None) -> Image.Image:
    if decoded_bytes is None:
        decoded_bytes, _ = _decode_base64_data(b64)
    return Image.open(io.BytesIO(decoded_bytes)).convert("RGB")


def _read_pdf_from_b64(b64: str, *, decoded_bytes: Optional[bytes] = None) -> List[Image.Image]:
    if decoded_bytes is None:
        decoded_bytes, _ = _decode_base64_data(b64)
    try:
        return convert_from_bytes(decoded_bytes)
    except Exception as exc:
        raise ValueError(f"PDF 转图片失败: {exc}")


def _collect_lines(result: Any, page: int) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    if not result:
        return lines
    for res in result:
        data = res
        if isinstance(data, dict) and "res" in data:
            data = data["res"]
        elif hasattr(data, "res"):
            data = data.res
        if not hasattr(data, "get") and hasattr(data, "items"):
            data = dict(data)
        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores") or []
        boxes = (
            data.get("rec_polys")
            or data.get("rec_boxes")
            or data.get("dt_polys")
            or []
        )

        for idx, text in enumerate(texts):
            if not text or not text.strip():
                continue
            score = float(scores[idx]) if idx < len(scores) else 0.0
            box = None
            if isinstance(boxes, (list, tuple)) and idx < len(boxes):
                box = boxes[idx]
            elif (
                hasattr(boxes, "__getitem__")
                and hasattr(boxes, "__len__")
                and idx < len(boxes)
            ):
                box = boxes[idx]
            if box is not None:
                box = np.asarray(box).tolist()
            lines.append({"text": text.strip(), "score": score, "box": box, "page": page})
    return lines

@app.post("/ocr", response_model=OcrResp)
def do_ocr(req: OcrReq):
    try:
        payload = req.file_b64 or req.image_b64
        assert payload is not None  # root_validator 已保证

        decoded_bytes, mime_hint = _decode_base64_data(payload)
        declared_type = (req.fileType or "").strip().lower()
        is_pdf = declared_type == "pdf"
        if not is_pdf:
            is_pdf = _is_pdf_b64(payload, decoded_bytes=decoded_bytes, mime_hint=mime_hint)

        if is_pdf:
            t0 = time.time()
            try:
                pages = _read_pdf_from_b64(payload, decoded_bytes=decoded_bytes)
            except Exception as exc:
                err = f"PDF 解析失败: {exc}"
                print(f"[ERR][PDF] {err}", file=sys.stderr, flush=True)
                return OcrResp(ok=False, lines=[], error=err)

            if not pages:
                err = "PDF 未解析到任何页面"
                print(f"[ERR][PDF] {err}", file=sys.stderr, flush=True)
                return OcrResp(ok=False, lines=[], error=err)

            total_lines: List[OcrLine] = []
            pages_resp: List[OcrPage] = []
            for page_idx, page_image in enumerate(pages, start=1):
                arr = np.array(page_image.convert("RGB"))
                try:
                    result = ocr.predict(arr, use_textline_orientation=USE_CLS)
                except Exception as exc:
                    err = f"第 {page_idx} 页 OCR 失败: {exc}"
                    print(f"[ERR][PDF] {err}", file=sys.stderr, flush=True)
                    return OcrResp(ok=False, lines=[], error=err)
                raw_lines = _collect_lines(result, page_idx)
                ocr_lines = [OcrLine(**l) for l in raw_lines]
                total_lines.extend(ocr_lines)
                page_text = "\n".join([line.text for line in ocr_lines if line.text]).strip()
                pages_resp.append(OcrPage(page=page_idx, lines=ocr_lines, text=page_text))

            dt = time.time() - t0
            print(
                f"[REQ] pdf pages={len(pages)} OK in {dt:.2f}s | device={'GPU' if USING_GPU else 'CPU'} | cls={USE_CLS}",
                file=sys.stderr,
                flush=True,
            )
            return OcrResp(ok=True, lines=total_lines, pages=pages_resp)

        # 默认按图片处理
        img = _read_image_from_b64(payload, decoded_bytes=decoded_bytes)
        arr = np.array(img)  # HWC, RGB
        t0 = time.time()
        result = ocr.predict(arr, use_textline_orientation=USE_CLS)
        dt = time.time() - t0
        print(
            f"[REQ] one image OK in {dt:.2f}s | device={'GPU' if USING_GPU else 'CPU'} | cls={USE_CLS}",
            file=sys.stderr,
            flush=True,
        )
        lines = _collect_lines(result, 1)
        ocr_lines = [OcrLine(**l) for l in lines]
        page_text = "\n".join([line.text for line in ocr_lines if line.text]).strip()
        return OcrResp(ok=True, lines=ocr_lines, pages=[OcrPage(page=1, lines=ocr_lines, text=page_text)])
    except Exception as e:
        err = str(e)
        print(f"[ERR] {err}", file=sys.stderr, flush=True)
        return OcrResp(ok=False, lines=[], error=err)

if __name__ == "__main__":
    print(f"[BOOT] prefer={PREF_DEV} | USING_GPU={USING_GPU} | cls={USE_CLS} | port={PORT}", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
