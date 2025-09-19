#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 优先的本地 OCR HTTP 服务（PaddleOCR 3.x）
- 优先使用 GPU（device="gpu"），若 cuDNN/驱动不可用会自动回退到 CPU（可用 OCR_GPU_STRICT=1 禁止回退）
- 预热：启动时做一次小图推理以加载模型，避免首个请求卡顿
- 路由：
    POST /ocr      { "image_b64": "<base64或dataURL>" } -> { ok, lines:[{text, score, box}], error? }
    GET  /healthz  -> { ok: true }
    GET  /version  -> { paddleocr, paddlepaddle, device, use_gpu, cls }
- 依赖：fastapi uvicorn pillow numpy paddleocr（以及已装好的 paddlepaddle-gpu 3.x + cuDNN8）
建议：
    pip install -U fastapi uvicorn pillow numpy
    pip install paddleocr>=3.0.0
    pip install "paddlepaddle-gpu==2.6.1" -f https://www.paddlepaddle.org.cn/whl/cu121.html
"""

import base64, io, os, sys, time, json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image, ImageDraw
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
# 识别语言（参考 https://github.com/PaddlePaddle/PaddleOCR/blob/release/3.0/doc/doc_ch/recognition_language.md ）
def _normalize_lang_code(lang: str) -> Tuple[str, Optional[str]]:
    """兼容常见写法，将语言代码规整到 PaddleOCR 支持的形式"""

    normalized = lang.strip().lower()
    if not normalized:
        return "ch", None

    alias = {
        "zh-cn": "ch",
        "zh_cn": "ch",
        "zh-hans": "ch",
        "zh_hans": "ch",
        "chs": "ch",
        "simplified": "ch",
        "zh-tw": "chinese_cht",
        "zh_tw": "chinese_cht",
        "zh-hk": "chinese_cht",
        "zh_hk": "chinese_cht",
        "zh-hant": "chinese_cht",
        "zh_hant": "chinese_cht",
        "cht": "chinese_cht",
        "ch_tra": "chinese_cht",
        "traditional": "chinese_cht",
        "chinese_traditional": "chinese_cht",
    }

    mapped = alias.get(normalized, normalized)
    if mapped != normalized:
        return mapped, mapped
    if normalized != lang:
        return normalized, normalized
    return normalized, None


OCR_LANG_RAW = os.getenv("OCR_LANG", "ch")
OCR_LANG, OCR_LANG_ALIAS = _normalize_lang_code(OCR_LANG_RAW)

if OCR_LANG_ALIAS:
    print(
        f"[CONFIG] PaddleOCR lang={OCR_LANG_RAW} -> {OCR_LANG}",
        file=sys.stderr,
        flush=True,
    )
else:
    print(f"[CONFIG] PaddleOCR lang={OCR_LANG}", file=sys.stderr, flush=True)

_T2S_CONVERTER = None
_OPENCC_T2S_ENABLED = OCR_LANG == "chinese_cht"
if _OPENCC_T2S_ENABLED:
    try:
        from opencc import OpenCC  # type: ignore

        _T2S_CONVERTER = OpenCC("t2s").convert
        print(
            "[CONFIG] Traditional Chinese output will be converted to Simplified via opencc.",
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        try:
            from zhconv import convert  # type: ignore

            _T2S_CONVERTER = lambda text: convert(text, "zh-hans")
            print(
                "[CONFIG] Traditional Chinese output will be converted to Simplified via zhconv.",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc2:  # pragma: no cover - optional dependency
            print(
                "[CONFIG][WARN] No T2S converter available (opencc error: "
                f"{exc}; zhconv error: {exc2}). Traditional text will remain unchanged.",
                file=sys.stderr,
                flush=True,
            )
            _OPENCC_T2S_ENABLED = False

# ================== 初始化与回退 ==================
def init_ocr_prefer_gpu() -> (PaddleOCR, bool):
    """优先 GPU 初始化；失败时（如 cuDNN 缺失）自动回退到 CPU，除非 GPU_STRICT=1"""
    # 优先尝试 GPU
    if PREF_DEV.startswith("gpu") or PREF_DEV == "auto":
        try:
            gpu_device = "gpu" if PREF_DEV == "auto" else PREF_DEV
            print(f"[INIT] Try GPU (device={gpu_device})...", file=sys.stderr, flush=True)
            ocr = PaddleOCR(
                lang=OCR_LANG,
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
        lang=OCR_LANG,
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
    image_b64: str  # 支持纯 base64 或 dataURL（data:image/...;base64,xxxx）

class OcrLine(BaseModel):
    text: str
    score: float
    box: List[List[float]]

class OcrResp(BaseModel):
    ok: bool
    lines: List[OcrLine] = []
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
            "lang": OCR_LANG,
        }
    except Exception:
        return {
            "paddleocr": "unknown",
            "paddlepaddle": "unknown",
            "device": "gpu" if USING_GPU else "cpu",
            "use_gpu": USING_GPU,
            "cls": USE_CLS,
            "lang": OCR_LANG,
        }

def _pdf_bytes_to_image_first_page(raw: bytes) -> Tuple[Image.Image, int]:
    import fitz  # type: ignore

    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        total_pages = doc.page_count or 0
        if total_pages <= 0:
            raise ValueError("PDF 文件没有可用页面")
        page = doc.load_page(0)
        pix = page.get_pixmap()
        if pix.alpha:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img, total_pages
    finally:
        doc.close()


def _read_image_from_b64(b64: str) -> Image.Image:
    # 兼容 dataURL
    if b64.strip().lower().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    header = raw.lstrip()
    if header.startswith(b"%PDF"):
        try:
            img, total_pages = _pdf_bytes_to_image_first_page(raw)
        except ImportError as exc:
            raise RuntimeError("处理 PDF 需要先安装 PyMuPDF：pip install PyMuPDF") from exc
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        print(f"[REQ] PDF base64 输入 -> 第 1 页 (共 {total_pages} 页)", file=sys.stderr, flush=True)
        return img
    return Image.open(io.BytesIO(raw)).convert("RGB")

def _locate_bottom_text_region(arr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """粗略估计图像中底部文字块的矩形区域."""

    if arr.ndim != 3 or arr.shape[2] < 3:
        return None

    h, w = arr.shape[:2]
    if h < 10 or w < 10:
        return None

    # 灰度 -> 统计每行深色像素数量，估计文字分布
    gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
    dark = gray < 200  # 允许老旧扫描件，阈值稍高
    row_density = dark.sum(axis=1)
    if not np.any(row_density):
        return None

    density_threshold = max(int(np.percentile(row_density, 80)), int(0.05 * w))
    candidate_rows = np.where(row_density >= density_threshold)[0]
    if candidate_rows.size == 0:
        return None

    # 聚焦于最底部的连通行段
    bottom_row = candidate_rows[-1]
    top_row = bottom_row
    for idx in range(candidate_rows.size - 1, 0, -1):
        if candidate_rows[idx] - candidate_rows[idx - 1] > 1:
            break
        top_row = candidate_rows[idx - 1]

    top_row = max(0, top_row - int(0.02 * h))
    bottom_row = min(h, bottom_row + int(0.02 * h) + 1)

    col_slice = dark[top_row:bottom_row, :].sum(axis=0)
    if not np.any(col_slice):
        return None

    col_threshold = max(int(np.percentile(col_slice, 70)), int(0.03 * h))
    cols = np.where(col_slice >= col_threshold)[0]
    if cols.size == 0:
        return None

    left_col = max(0, cols[0] - int(0.01 * w))
    right_col = min(w, cols[-1] + int(0.01 * w) + 1)

    if right_col - left_col < 5 or bottom_row - top_row < 5:
        return None

    return left_col, top_row, right_col, bottom_row


@app.post("/ocr", response_model=OcrResp)
def do_ocr(req: OcrReq):
    try:
        img = _read_image_from_b64(req.image_b64)
        arr = np.array(img)  # HWC, RGB

        crop_box = _locate_bottom_text_region(arr)
        if crop_box:
            x0, y0, x1, y1 = crop_box
            arr_for_ocr = arr[y0:y1, x0:x1]
            print(
                f"[CROP] bottom text region located at (x0={x0}, y0={y0}, x1={x1}, y1={y1})",
                file=sys.stderr,
                flush=True,
            )
        else:
            arr_for_ocr = arr
            x0 = y0 = 0

        t0 = time.time()
        result = ocr.predict(arr_for_ocr, use_textline_orientation=USE_CLS)
        dt = time.time() - t0
        print(f"[REQ] one image OK in {dt:.2f}s | device={'GPU' if USING_GPU else 'CPU'} | cls={USE_CLS}",
              file=sys.stderr, flush=True)

        lines: List[Dict[str, Any]] = []
        if result:
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
                    text_clean = text.strip()
                    converted = text_clean
                    if _OPENCC_T2S_ENABLED and _T2S_CONVERTER is not None:
                        try:
                            converted = _T2S_CONVERTER(text_clean)
                        except Exception as exc:  # pragma: no cover - conversion failure shouldn't break OCR
                            print(
                                f"[REC][WARN] T2S convert failed: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )
                        else:
                            print(
                                f"[REC][T2S] '{text_clean}' -> '{converted}'",
                                file=sys.stderr,
                                flush=True,
                            )
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
                        box_arr = np.asarray(box, dtype=float)
                        if crop_box:
                            offset = np.array([[x0, y0]], dtype=box_arr.dtype)
                            if box_arr.ndim == 2:
                                box_arr = box_arr + offset
                            else:
                                box_arr = box_arr + offset.reshape((-1, 1, 2))
                        box = box_arr.tolist()
                    final_text = converted
                    print(
                        f"[REC] text='{final_text}' score={score:.4f}",
                        file=sys.stderr,
                        flush=True,
                    )
                    lines.append({"text": final_text, "score": score, "box": box})
        return OcrResp(ok=True, lines=[OcrLine(**l) for l in lines])
    except Exception as e:
        err = str(e)
        print(f"[ERR] {err}", file=sys.stderr, flush=True)
        return OcrResp(ok=False, lines=[], error=err)

if __name__ == "__main__":
    print(f"[BOOT] prefer={PREF_DEV} | USING_GPU={USING_GPU} | cls={USE_CLS} | port={PORT}", file=sys.stderr)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
