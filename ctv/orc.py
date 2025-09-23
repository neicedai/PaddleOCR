#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 优先的本地 OCR HTTP 服务（PaddleOCR 3.x）
- 优先使用 GPU（device="gpu"），若 cuDNN/驱动不可用会自动回退到 CPU（可用 OCR_GPU_STRICT=1 禁止回退）
- 预热：启动时做一次小图推理以加载模型，避免首个请求卡顿
- 路由：
    POST /ocr      { "image_b64": "<base64或dataURL>", "target_script"?:"auto|simplified|traditional" }
                  -> { ok, lines:[{text, score, box}], error? }
    GET  /healthz  -> { ok: true }
    GET  /version  -> { paddleocr, paddlepaddle, device, use_gpu, cls }
- 依赖：fastapi uvicorn pillow numpy paddleocr（以及已装好的 paddlepaddle-gpu 3.x + cuDNN8）
建议：
    pip install -U fastapi uvicorn pillow numpy
    pip install paddleocr>=3.0.0
    pip install "paddlepaddle-gpu==2.6.1" -f https://www.paddlepaddle.org.cn/whl/cu121.html
- 当 OCR_LANG 选择中文模型时会自动启用繁体->简体转换，可用 OCR_T2S=0 强制关闭，也可用 OCR_T2S=1 强制开启
"""

import base64, io, os, sys, time, json
from typing import List, Dict, Any, Optional, Tuple, Literal

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
# 识别结果分数阈值（低于该分数的文本将被丢弃，<=0 表示不过滤）
try:
    REC_SCORE_MIN = float(os.getenv("OCR_REC_SCORE_MIN", "0.3"))
except ValueError:
    print(
        "[CONFIG][WARN] Invalid OCR_REC_SCORE_MIN value, fallback to 0.3",
        file=sys.stderr,
        flush=True,
    )
    REC_SCORE_MIN = 0.3
# 强制设备（仅用于日志提示）：gpu / cpu（实际设备由 use_gpu & 环境决定）
PREF_DEV     = os.getenv("OCR_DEVICE", "gpu").lower()
# 繁->简转换（缺省对中文模型自动开启，OCR_T2S=0/1 可强制关闭/开启）
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


def _parse_env_toggle(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized or normalized == "auto":
        return None
    if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return None


_T2S_CONVERTER = None
_t2s_override = _parse_env_toggle(os.getenv("OCR_T2S"))
_lang_is_chinese = OCR_LANG == "ch" or OCR_LANG.startswith("chinese")
_OPENCC_T2S_ENABLED = (
    _t2s_override if _t2s_override is not None else _lang_is_chinese
)

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

if _T2S_CONVERTER is not None:
    if _t2s_override is None:
        print(
            f"[CONFIG] Simplified logging enabled automatically for lang={OCR_LANG}.",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "[CONFIG] Simplified logging enabled via OCR_T2S toggle.",
            file=sys.stderr,
            flush=True,
        )
elif _t2s_override is False:
    print(
        "[CONFIG] Simplified logging disabled via OCR_T2S toggle.",
        file=sys.stderr,
        flush=True,
    )
elif not _lang_is_chinese:
    print(
        "[CONFIG] Simplified logging disabled (non-Chinese model).",
        file=sys.stderr,
        flush=True,
    )
else:
    print(
        "[CONFIG] Simplified logging unavailable (converter missing).",
        file=sys.stderr,
        flush=True,
    )

print(
    f"[CONFIG] recognition score threshold={REC_SCORE_MIN}",
    file=sys.stderr,
    flush=True,
)

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
    """HTTP OCR 请求体.

    ``page_index`` 为可选的 1-based 页码；当客户端将其设置为 ``1`` 时，
    服务会跳过底部聚类逻辑，以完整保留封面页。

    ``target_script`` 控制返回文本的脚本风格：

    - ``"auto"``（默认）会在检测到繁体时尝试转换成简体；若无差异则保持原文；
    - ``"simplified"`` 始终尝试繁转简；
    - ``"traditional"`` 保留原始识别结果。
    """

    image_b64: str  # 支持纯 base64 或 dataURL（data:image/...;base64,xxxx）
    page_index: Optional[int] = None
    target_script: Literal["auto", "simplified", "traditional"] = "auto"

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

def _cluster_bottom_text_entries(
    entries: List[Tuple[int, np.ndarray]], image_height: int
) -> Optional[List[int]]:
    """基于检测框的纵向位置，将最底部的一段文字行聚为一个簇.

    返回属于底部簇的 entry 索引列表，若无法确定则返回 ``None``。
    """

    if not entries or image_height <= 0:
        return None

    # 收集每个文本框的 (top_y, bottom_y) 范围，并按照 top_y 升序排序
    spans: List[Tuple[int, float, float]] = []
    for idx, box in entries:
        if box.size == 0:
            continue
        ys = box[..., 1].reshape(-1)
        spans.append((idx, float(ys.min()), float(ys.max())))

    if not spans:
        return None

    spans.sort(key=lambda item: item[1])
    bottom_idx, bottom_top, bottom_bottom = spans[-1]
    selected = {bottom_idx}
    cluster_top = bottom_top
    cluster_bottom = bottom_bottom

    # 从底部向上合并间隔较小的文本行，允许轻微的留白
    gap_tolerance = max(image_height * 0.02, 8.0)
    for idx, top, bottom in reversed(spans[:-1]):
        if cluster_top - bottom > gap_tolerance:
            break
        selected.add(idx)
        cluster_top = min(cluster_top, top)
        cluster_bottom = max(cluster_bottom, bottom)

    # 若聚合后的底部文本并不靠近图像底部，则认为判断不可靠
    if cluster_bottom < image_height * 0.45:
        return None

    # 将所有与该纵向范围有重叠的行都纳入，避免遗漏同一文本块
    for idx, top, bottom in spans:
        if idx in selected:
            continue
        overlaps = not (bottom < cluster_top or top > cluster_bottom)
        if overlaps:
            selected.add(idx)

    return sorted(selected)


@app.post("/ocr", response_model=OcrResp)
def do_ocr(req: OcrReq):
    try:
        img = _read_image_from_b64(req.image_b64)
        arr = np.array(img)  # HWC, RGB

        target_script = req.target_script or "auto"
        print(
            f"[REQ] target_script={target_script} | converter={'yes' if _T2S_CONVERTER else 'no'}",
            file=sys.stderr,
            flush=True,
        )

        t0 = time.time()
        result = ocr.predict(arr, use_textline_orientation=USE_CLS)
        dt = time.time() - t0
        print(f"[REQ] one image OK in {dt:.2f}s | device={'GPU' if USING_GPU else 'CPU'} | cls={USE_CLS}",
              file=sys.stderr, flush=True)

        parsed_entries: List[Dict[str, Any]] = []
        box_entries: List[Tuple[int, np.ndarray]] = []
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
                    original_text = text.strip()
                    score = float(scores[idx]) if idx < len(scores) else 0.0
                    if REC_SCORE_MIN > 0 and score < REC_SCORE_MIN:
                        print(
                            f"[REC][SKIP] text='{original_text}' score={score:.4f} < {REC_SCORE_MIN}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    box_raw = None
                    if isinstance(boxes, (list, tuple)) and idx < len(boxes):
                        box_raw = boxes[idx]
                    elif (
                        hasattr(boxes, "__getitem__")
                        and hasattr(boxes, "__len__")
                        and idx < len(boxes)
                    ):
                        box_raw = boxes[idx]
                    box_arr = None
                    if box_raw is not None:
                        box_arr = np.asarray(box_raw, dtype=float)
                        box_entries.append((len(parsed_entries), box_arr))
                    parsed_entries.append({
                        "raw_text": original_text,
                        "score": score,
                        "box_arr": box_arr,
                    })

        selected_indices: Optional[List[int]] = None
        is_cover_page = req.page_index == 1 if req.page_index is not None else False
        if is_cover_page:
            print(
                f"[CROP] page_index={req.page_index} -> cover detected, use full-page mode",
                file=sys.stderr,
                flush=True,
            )
        elif parsed_entries:
            selected_indices = _cluster_bottom_text_entries(box_entries, arr.shape[0])
            if selected_indices:
                print(
                    f"[CROP] focus bottom lines idx={selected_indices}",
                    file=sys.stderr,
                    flush=True,
                )

        lines: List[Dict[str, Any]] = []
        indices_to_keep = selected_indices or list(range(len(parsed_entries)))
        for idx in indices_to_keep:
            entry = parsed_entries[idx]
            raw_text = entry["raw_text"]
            final_text = raw_text
            if target_script == "traditional":
                pass
            elif _T2S_CONVERTER is None:
                if target_script == "simplified":
                    print(
                        "[REC][SCRIPT] target=simplified but converter unavailable; keep traditional text",
                        file=sys.stderr,
                        flush=True,
                    )
            else:
                try:
                    converted_text = _T2S_CONVERTER(raw_text)
                    if converted_text is None:
                        converted_text = raw_text
                except Exception as exc:  # pragma: no cover
                    print(
                        f"[REC][WARN] T2S convert failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    if target_script == "simplified":
                        final_text = converted_text
                        if converted_text != raw_text:
                            print(
                                f"[REC][T2S] target=simplified '{raw_text}' -> '{converted_text}'",
                                file=sys.stderr,
                                flush=True,
                            )
                        else:
                            print(
                                f"[REC][SCRIPT] target=simplified no change for '{raw_text}'",
                                file=sys.stderr,
                                flush=True,
                            )
                    elif target_script == "auto" and converted_text != raw_text:
                        final_text = converted_text
                        print(
                            f"[REC][T2S] target=auto '{raw_text}' -> '{converted_text}'",
                            file=sys.stderr,
                            flush=True,
                        )
                    elif target_script == "auto":
                        print(
                            f"[REC][SCRIPT] target=auto no change for '{raw_text}'",
                            file=sys.stderr,
                            flush=True,
                        )
            score = entry["score"]
            box_arr = entry["box_arr"]
            box = box_arr.tolist() if box_arr is not None else None
            entry["final_text"] = final_text
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
