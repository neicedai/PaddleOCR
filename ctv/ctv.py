#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ctv_cloud_env.py  (with iFLYTEK Private WS TTS integrated)
连环画 -> 云OCR/本地OCR + 多TTS -> 极速视频

- OCR：腾讯/百度/RapidOCR/Tesseract/HTTP；整图识别；清洗（去 www.lhhl.com / 行首(数字) / “向日葵连环画”）
- TTS：edge（兼容旧edge-tts：不传pitch）、azure、elevenlabs、gtts、pyttsx3、iflytek(私有域WS)、HTTP
- iflytek（你给的协议）：鉴权=HMAC-SHA256；x5_* 自动忽略 parameter.oral；支持 by-sent 用 [pXXX] 停顿
- 跳过页：--skip-stems 仅生成静音，不做 OCR/TTS
- 朗读：按句切分更自然
- 快速管线：分段 ffmpeg 编码 + 无重编码 concat； MoviePy 2.x 兼容
"""

import argparse, os, re, sys, tempfile, subprocess, base64, time, json, asyncio, ssl, hashlib, hmac, urllib.parse, zipfile, shutil
from contextlib import ExitStack
from io import BytesIO
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Sequence, Literal
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment
from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips, vfx
import requests
from dotenv import load_dotenv


def _bool_from(value: Optional[Any], default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return text not in {"0", "false", "no", "off"}


def _maybe_int(value: Optional[Any]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Optional[Any]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _env_for_emotion(key: str, emotion: Optional[str]) -> Optional[str]:
    if not key:
        return None
    if emotion:
        emo_key = f"{key}_{emotion.upper()}"
        if emo_key in os.environ:
            return os.getenv(emo_key)
    return os.getenv(key)

# ---------------- 基础工具 ----------------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images_sorted(images_dir: Path) -> List[Path]:
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    return files


def list_pdfs_sorted(path: Path) -> List[Path]:
    pdfs = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    pdfs.sort(key=lambda p: natural_key(p.name))
    return pdfs


def convert_pdf_to_images(pdf_path: Path, output_dir: Path) -> List[Path]:
    import fitz  # type: ignore

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    try:
        img_paths: List[Path] = []
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            pix = page.get_pixmap()
            if pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_path = output_dir / f"{pdf_path.stem}_{page_idx + 1:04d}.png"
            pix.save(str(img_path))
            img_paths.append(img_path)
        return img_paths
    finally:
        doc.close()

def ensure_font(size: int):
    cands = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for fp in cands:
        if os.path.exists(fp):
            try: return ImageFont.truetype(fp, size)
            except: pass
    return ImageFont.load_default()

# ---- 文本清洗 + 分句 ----
def clean_text_for_manhua(raw: str) -> str:
    """
    - 删除行首的 (数字) / （数字）
    - 去除 www.lhhl.com（水印，不区分大小写）
    - 去除 “向日葵连环画” 文案（整行或行内）
    - 去空行
    """
    if not raw:
        return ""
    lines = []
    for line in raw.splitlines():
        line = re.sub(r'^\s*[\(（]\s*\d+\s*[\)）]\s*', '', line)
        line = re.sub(r'^\s*[\(（]\s*[＊*]+\s*[\)）]?\s*', '', line)
        line = re.sub(r'(?i)www\.lhhl\.com', '', line)
        line = re.sub(r'向日葵连环画', '', line)
        line = line.strip()
        if not line:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def filter_text_by_keywords(text: str, keywords: Sequence[str]) -> str:
    """Remove lines that contain any of the provided keywords."""
    if not text or not keywords:
        return text
    filtered_lines: List[str] = []
    for line in text.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if any(kw and kw in line_stripped for kw in keywords):
            continue
        filtered_lines.append(line_stripped)
    return "\n".join(filtered_lines).strip()

_SENT_PAT = re.compile(r'[^。！？!?；;：:…]+[。！？!?；;：:…]{0,2}(?:"|”|\'|’)?')

def to_sentence_list(cleaned: str) -> List[str]:
    """
    把多行合成全文，按中文标点切句；保留标点，去冗余空白
    """
    if not cleaned:
        return []
    text = re.sub(r'\s*\n+\s*', '', cleaned)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    sents = [s.strip() for s in _SENT_PAT.findall(text)]
    sents = [s for s in sents if s]
    if not sents:
        return [text]

    # 若切分后的句子拼接无法还原原文，则进行回退，以避免丢字。
    compact_original = re.sub(r"\s+", "", text)
    compact_joined = re.sub(r"\s+", "", "".join(sents))
    if compact_original and compact_joined != compact_original:
        # 先尝试按常见标点拆分，若仍无法覆盖，则退回整段文本。
        naive = [seg.strip() for seg in re.split(r"(?<=[。！？!?；;：:…])", text) if seg.strip()]
        compact_naive = re.sub(r"\s+", "", "".join(naive)) if naive else ""
        if naive and compact_naive == compact_original:
            return naive
        return [text]
    return sents

def join_for_tts(sents: List[str], sep: str = ' ') -> str:
    return sep.join([s.strip() for s in sents if s.strip()])

# ---------------- OCR 预处理 ----------------
def _otsu_threshold(arr: np.ndarray) -> int:
    hist,_ = np.histogram(arr, bins=256, range=(0,256))
    total = arr.size; sum_total = np.dot(np.arange(256), hist)
    sumB=0; wB=0; maxv=0; thr=0
    for i in range(256):
        wB += hist[i]
        if wB==0: continue
        wF = total - wB
        if wF==0: break
        sumB += i*hist[i]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var = wB*wF*(mB-mF)**2
        if var>maxv: maxv=var; thr=i
    return thr

def preprocess_for_ocr(img: Image.Image,
                       crop_box: Optional[Tuple[float,float,float,float]],
                       strip_border_ratio: float,
                       scale: float,
                       binarize: bool) -> Image.Image:
    if crop_box:
        x1,y1,x2,y2 = crop_box
        W,H = img.size
        img = img.crop((int(W*x1), int(H*y1), int(W*x2), int(H*y2)))
    if strip_border_ratio and strip_border_ratio>0:
        W,H = img.size; b = int(min(W,H)*strip_border_ratio)
        if b>0 and (W-2*b)>10 and (H-2*b)>10:
            img = img.crop((b,b,W-b,H-b))
    if scale and abs(scale-1.0)>1e-3:
        W,H = img.size
        img = img.resize((int(W*scale), int(H*scale)), Image.LANCZOS)
    if binarize:
        g = img.convert("L"); a = np.array(g); thr = _otsu_threshold(a)
        a2 = (a>thr).astype(np.uint8)*255; img = Image.fromarray(a2).convert("L")
    return img

# ---------------- OCR 引擎 ----------------
def ocr_text_rapidocr(img: Image.Image) -> str:
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    arr = np.array(img.convert("RGB"))[:, :, ::-1]  # RGB->BGR
    result, _ = ocr(arr)
    lines = [item[1] for item in (result or []) if len(item)>=2]
    return "\n".join(lines).strip()

def ocr_text_tesseract(img: Image.Image, lang: str, psm: int, oem: int) -> str:
    import pytesseract
    cfg = f"--oem {oem} --psm {psm}"
    return pytesseract.image_to_string(img.convert("L"), lang=lang, config=cfg).strip()

_baidu_token_cache = {"token": None, "exp": 0}
def _baidu_get_token(ak: str, sk: str) -> str:
    now = time.time()
    if _baidu_token_cache["token"] and now < _baidu_token_cache["exp"] - 60:
        return _baidu_token_cache["token"]
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": ak, "client_secret": sk}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    _baidu_token_cache["token"] = data["access_token"]
    _baidu_token_cache["exp"] = now + int(data.get("expires_in", 2592000))
    return _baidu_token_cache["token"]

def ocr_text_baidu(img: Image.Image, ak: str, sk: str, accurate=True) -> str:
    token = _baidu_get_token(ak, sk)
    endpoint = "accurate_basic" if accurate else "general_basic"
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/{endpoint}?access_token={token}"
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {"image": img_b64, "language_type": "CHN_ENG", "paragraph": "true"}
    r = requests.post(url, data=payload, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    words = [it.get("words","").strip() for it in data.get("words_result", []) if it.get("words")]
    return "\n".join(words).strip()

def ocr_text_tencent(img: Image.Image, secret_id: str, secret_key: str,
                     region: str="ap-beijing", model: str="basic") -> str:
    from tencentcloud.common import credential
    from tencentcloud.ocr.v20181119 import ocr_client, models
    cred = credential.Credential(secret_id, secret_key)
    client = ocr_client.OcrClient(cred, region)
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    raw = buf.getvalue()
    img_b64 = base64.b64encode(raw).decode()
    if model == "accurate":
        req = models.GeneralAccurateOCRRequest()
        req.ImageBase64 = img_b64
        resp = client.GeneralAccurateOCR(req)
    else:
        req = models.GeneralBasicOCRRequest()
        req.ImageBase64 = img_b64
        resp = client.GeneralBasicOCR(req)
    text_lines = [d.DetectedText for d in resp.TextDetections]
    return "\n".join(text_lines).strip()

def ocr_text_http(
    img: Image.Image,
    url: str,
    page_index: Optional[int] = None,
    target_script: Literal["auto", "simplified", "traditional"] = "auto",
) -> str:
    """调用本地 HTTP OCR 服务，并允许选择输出脚本风格."""
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    payload: Dict[str, Any] = {"image_b64": img_b64}
    if page_index is not None:
        try:
            payload["page_index"] = int(page_index)
        except (TypeError, ValueError):
            pass
    normalized_script = (target_script or "auto").strip().lower()
    if normalized_script not in {"auto", "simplified", "traditional"}:
        raise ValueError(f"invalid target_script: {target_script}")
    if normalized_script != "auto":
        payload["target_script"] = normalized_script
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "OCR service error"))
    lines = [l.get("text", "") for l in data.get("lines", []) if l.get("text")]
    return "\n".join(lines).strip()

def ocr_image(image_path: Path,
              engine: str,
              lang: str, psm: int, oem: int,
              crop_box: Optional[Tuple[float,float,float,float]],
              strip_border: float, scale: float, binarize: bool,
              tencent_model: str, debug: bool=False,
              page_index: Optional[int] = None) -> str:
    img = Image.open(image_path)
    img = preprocess_for_ocr(img, crop_box, strip_border, scale, binarize)
    if engine == "rapidocr":
        text = ocr_text_rapidocr(img)
    elif engine == "tesseract":
        text = ocr_text_tesseract(img, lang, psm, oem)
    elif engine == "baidu":
        ak = os.getenv("BAIDU_OCR_AK"); sk = os.getenv("BAIDU_OCR_SK")
        if not ak or not sk: raise RuntimeError("BAIDU_OCR_AK/BAIDU_OCR_SK 未设置（.env）")
        text = ocr_text_baidu(img, ak, sk, accurate=True)
    elif engine == "tencent":
        sid = os.getenv("TENCENT_SECRET_ID"); skey = os.getenv("TENCENT_SECRET_KEY")
        region = os.getenv("TENCENT_REGION","ap-beijing")
        if not sid or not skey: raise RuntimeError("TENCENT_SECRET_ID/TENCENT_SECRET_KEY 未设置（.env）")
        text = ocr_text_tencent(img, sid, skey, region=region, model=tencent_model)
    elif engine == "http":
        url = os.getenv("OCR_API_URL", "http://127.0.0.1:8001/ocr")
        text = ocr_text_http(img, url, page_index=page_index)
    else:
        raise ValueError(f"未知 ocr-engine: {engine}")
    if debug:
        print(f"[OCR] {engine} OK -> {image_path.name} (model={tencent_model if engine=='tencent' else '-'}, chars={len(text)})")
    return text

# ---------------- TTS：Edge（不传 pitch；兼容旧 edge-tts） ----------------
async def _edge_tts_one(text: str, voice: str, rate: Optional[str], out_path: Path):
    import edge_tts
    kwargs = {"text": text, "voice": voice}
    if rate:
        kwargs["rate"] = rate
    com = edge_tts.Communicate(**kwargs)
    await com.save(str(out_path))

def save_audio_edge_tts_from_sentences(sentences, voice, rate, _pitch_ignored, _style_unused, out_path: Path, on_empty: str):
    sents = [s.strip() for s in (sentences or []) if s.strip()]
    if not sents:
        if on_empty == "silence":
            AudioSegment.silent(duration=600).export(out_path, format="mp3"); return
        sents = ["这一页没有可读文本"]
    tmp_files=[]
    try:
        for sent in sents:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                tmp_files.append(tf.name)
            asyncio.run(_edge_tts_one(sent, voice, rate, Path(tmp_files[-1])))
        pieces=[]; silence = AudioSegment.silent(duration=300)
        for i, fp in enumerate(tmp_files):
            seg = AudioSegment.from_file(fp)
            pieces.append(seg)
            if i != len(tmp_files)-1: pieces.append(silence)
        final = sum(pieces) if pieces else AudioSegment.silent(duration=600)
        final.export(out_path, format="mp3")
    finally:
        for fp in tmp_files:
            try: os.remove(fp)
            except: pass

# ---------------- TTS：Azure ----------------
def save_audio_azure_tts_from_sentences(sentences: List[str], voice: str, rate: str, pitch: str, style: Optional[str],
                                        out_path: Path, on_empty: str):
    key = os.getenv("AZURE_SPEECH_KEY"); region = os.getenv("AZURE_SPEECH_REGION")
    if not key or not region: raise RuntimeError("AZURE_SPEECH_KEY/AZURE_SPEECH_REGION 未设置（.env）")
    sents = [s.strip() for s in sentences if s.strip()]
    if not sents:
        if on_empty == "silence":
            AudioSegment.silent(duration=600).export(out_path, format="mp3"); return
        sents = ["这一页没有可读文本"]
    inner=[]
    for i,s in enumerate(sents):
        inner.append(f"<s>{s}</s>")
        if i!=len(sents)-1: inner.append('<break time="300ms"/>')
    inner_xml="\n      ".join(inner)
    body = (f'<mstts:express-as style="{style}" xmlns:mstts="http://www.w3.org/2001/mstts">{inner_xml}</mstts:express-as>'
            if style else inner_xml)
    ssml=f'''
<speak version="1.0" xml:lang="zh-CN">
  <voice name="{voice}">
    <prosody rate="{rate}" pitch="{pitch}">
      {body}
    </prosody>
  </voice>
</speak>'''.strip()
    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-24khz-160kbitrate-mono-mp3",
        "User-Agent": "ctv-cloud-env"
    }
    r = requests.post(url, data=ssml.encode("utf-8"), headers=headers, timeout=45)
    r.raise_for_status()
    Path(out_path).write_bytes(r.content)

# ---------------- TTS：ElevenLabs ----------------
def save_audio_elevenlabs(text: str, voice_id: str, out_path: Path, on_empty: str):
    api_key = os.getenv("ELEVEN_API_KEY")
    if not api_key: raise RuntimeError("ELEVEN_API_KEY 未设置（.env）")
    t = text.strip()
    if not t:
        if on_empty == "silence":
            AudioSegment.silent(duration=600).export(out_path, format="mp3"); return
        t = "这一页没有可读文本"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {"xi-api-key": api_key, "accept": "audio/mpeg", "content-type": "application/json"}
    payload = {"text": t, "model_id": "eleven_multilingual_v2"}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    Path(out_path).write_bytes(r.content)

# ---------------- TTS：gTTS / pyttsx3 ----------------
def save_audio_gtts_from_sentences(sentences: List[str], voice_lang: str, out_path: Path, on_empty: str):
    from gtts import gTTS
    sents = [s.strip() for s in sentences if s.strip()]
    if sents:
        text = join_for_tts(sents, sep=' ')
        gTTS(text=text, lang=voice_lang).save(str(out_path))
    else:
        if on_empty == "silence":
            AudioSegment.silent(duration=600).export(out_path, format="mp3")
        else:
            gTTS(text="这一页没有可读文本", lang=voice_lang).save(str(out_path))

def save_audio_pyttsx3_from_sentences(sentences: List[str], voice_name: Optional[str], out_path: Path, on_empty: str):
    import pyttsx3
    sents = [s.strip() for s in sentences if s.strip()]
    if not sents and on_empty=="silence":
        AudioSegment.silent(duration=600).export(out_path, format="mp3"); return
    text = join_for_tts(sents if sents else ["这一页没有可读文本"], sep=' ')
    eng = pyttsx3.init()
    if voice_name:
        for v in eng.getProperty("voices"):
            if voice_name.lower() in (getattr(v,"name","") or "").lower():
                eng.setProperty("voice", v.id); break
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpwav:
        tmp = tmpwav.name
    try:
        eng.save_to_file(text, tmp); eng.runAndWait()
        seg = AudioSegment.from_wav(tmp); seg.export(out_path, format="mp3")
    finally:
        try: os.remove(tmp)
        except: pass

def save_audio_http_tts_from_sentences(sentences: List[str], api_url: str,
                                      voice_lang: str, voice_name: Optional[str], speed: float,
                                      out_path: Path, on_empty: str):
    sents = [s.strip() for s in sentences if s.strip()]
    if not sents:
        if on_empty == "silence":
            AudioSegment.silent(duration=600).export(out_path, format="mp3"); return
        sents = ["这一页没有可读文本"]
    text = join_for_tts(sents, sep=' ')
    payload: Dict[str, Any] = {"text": text}
    if voice_lang: payload["language"] = voice_lang
    if voice_name: payload["speaker"] = voice_name
    if speed is not None: payload["speed"] = speed
    r = requests.post(api_url, json=payload, timeout=60)
    r.raise_for_status()
    if r.headers.get("content-type", "").startswith("audio"):
        seg = AudioSegment.from_file(BytesIO(r.content))
        seg.export(out_path, format="mp3")
    else:
        try:
            data = r.json()
            raise RuntimeError(data.get("error", "TTS service error"))
        except Exception:
            raise RuntimeError("TTS service error")


def save_audio_chatts_from_sentences(sentences: List[str], api_url: str,
                                    prompt: Optional[str], speaker: Optional[str], seed: Optional[int],
                                    lang: Optional[str], enable_refine: bool,
                                    refine_prompt: Optional[str], refine_seed: Optional[int],
                                    do_normalize: bool, do_homophone: bool,
                                    timeout: float, out_path: Path, on_empty: str):
    sents = [s.strip() for s in sentences if s.strip()]
    if not sents:
        if on_empty == "silence":
            AudioSegment.silent(duration=600).export(out_path, format="mp3")
            return
        sents = ["这一页没有可读文本"]

    payload: Dict[str, Any] = {
        "text": sents,
        "stream": False,
        "skip_refine_text": not enable_refine,
        "refine_text_only": False,
        "use_decoder": True,
        "do_text_normalization": bool(do_normalize),
        "do_homophone_replacement": bool(do_homophone),
    }
    if lang:
        payload["lang"] = lang

    params_infer: Dict[str, Any] = {"prompt": prompt or "[speed_5]"}
    if seed is not None:
        params_infer["manual_seed"] = int(seed)
    if speaker:
        params_infer["spk_emb"] = speaker
    payload["params_infer_code"] = params_infer

    if enable_refine:
        params_refine: Dict[str, Any] = {"prompt": refine_prompt or ""}
        if refine_seed is not None:
            params_refine["manual_seed"] = int(refine_seed)
        payload["params_refine_text"] = params_refine

    resp = requests.post(api_url, json=payload, timeout=timeout)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "").lower()
    raw = resp.content
    segments: List[AudioSegment] = []
    silence = AudioSegment.silent(duration=300)

    if raw.startswith(b"PK") or "zip" in content_type:
        with zipfile.ZipFile(BytesIO(raw)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a'))]
            if not names:
                raise RuntimeError("ChatTTS API 未返回音频文件")
            names.sort()
            for idx, name in enumerate(names):
                data = zf.read(name)
                seg = AudioSegment.from_file(BytesIO(data))
                segments.append(seg)
                if idx != len(names) - 1:
                    segments.append(silence)
    else:
        seg = AudioSegment.from_file(BytesIO(raw))
        segments.append(seg)

    if not segments:
        raise RuntimeError("ChatTTS API 未返回有效音频数据")
    final_seg = segments[0]
    for seg in segments[1:]:
        final_seg += seg
    final_seg.export(out_path, format="mp3")


def save_audio_indextts_from_sentences(
    sentences: List[str],
    api_url: str,
    prompt_wav: str,
    emo_wav: Optional[str],
    emo_text: Optional[str],
    use_emo_text: bool,
    emo_alpha: float,
    emo_vector_json: Optional[str],
    use_random: bool,
    interval_silence: Optional[int],
    max_text_tokens: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    repetition_penalty: Optional[float],
    max_mel_tokens: Optional[int],
    timeout: float,
    out_path: Path,
    on_empty: str,
):
    sents = [s.strip() for s in sentences if s.strip()]
    if not sents:
        if on_empty == "silence":
            AudioSegment.silent(duration=600).export(out_path, format="mp3")
            return
        sents = ["这一页没有可读文本"]

    if not api_url:
        raise RuntimeError("缺少 IndexTTS API 地址（INDEXTTS_API_URL）")
    if not prompt_wav or not os.path.exists(prompt_wav):
        raise RuntimeError("缺少 IndexTTS 参考音频（INDEXTTS_PROMPT_WAV）")
    emo_wav = emo_wav.strip() if emo_wav else None
    if emo_wav:
        if not os.path.exists(emo_wav):
            raise RuntimeError(f"IndexTTS 情感参考音频不存在：{emo_wav}")

    text = join_for_tts(sents, sep=' ')

    data: Dict[str, Any] = {
        "text": text,
        "use_emo_text": "true" if use_emo_text else "false",
        "emo_alpha": str(float(emo_alpha if emo_alpha is not None else 1.0)),
        "use_random": "true" if use_random else "false",
    }
    if emo_text:
        data["emo_text"] = emo_text
    if emo_vector_json:
        data["emo_vector_json"] = emo_vector_json
    if interval_silence is not None:
        data["interval_silence"] = str(int(interval_silence))
    if max_text_tokens is not None:
        data["max_text_tokens_per_segment"] = str(int(max_text_tokens))
    if temperature is not None:
        data["temperature"] = str(float(temperature))
    if top_p is not None:
        data["top_p"] = str(float(top_p))
    if top_k is not None:
        data["top_k"] = str(int(top_k))
    if repetition_penalty is not None:
        data["repetition_penalty"] = str(float(repetition_penalty))
    if max_mel_tokens is not None:
        data["max_mel_tokens"] = str(int(max_mel_tokens))

    prompt_name = os.path.basename(prompt_wav) or "prompt.wav"
    files: Dict[str, Any] = {}
    with ExitStack() as stack:
        prompt_f = stack.enter_context(open(prompt_wav, "rb"))
        files["speaker_audio"] = (prompt_name, prompt_f, "audio/wav")
        if emo_wav:
            emo_name = os.path.basename(emo_wav) or "emo.wav"
            emo_f = stack.enter_context(open(emo_wav, "rb"))
            files["emo_audio"] = (emo_name, emo_f, "audio/wav")

        try:
            response = requests.post(api_url, data=data, files=files, timeout=max(timeout, 1.0))
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"IndexTTS 请求失败: {exc}")

    content_type = response.headers.get("content-type", "")
    if content_type.lower().startswith("audio"):
        seg = AudioSegment.from_file(BytesIO(response.content))
        seg.export(out_path, format="mp3")
        return

    try:
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error")
        if detail:
            raise RuntimeError(f"IndexTTS API 错误: {detail}")
    raise RuntimeError("IndexTTS API 未返回音频数据")


# ---------------- TTS：iFLYTEK 私有域 WS ----------------
def _rfc1123_date():
    import datetime
    return datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")

def _ifly_sign_ws_url(raw_ws_url: str, api_key: str, api_secret: str) -> str:
    u = urlparse(raw_ws_url)
    if u.scheme not in ("ws","wss"):
        raise ValueError(f"无效的 WS URL: {raw_ws_url}")
    host = u.netloc
    path = u.path or "/"
    date = _rfc1123_date()
    signature_origin = f"host: {host}\n" + f"date: {date}\n" + f"GET {path} HTTP/1.1"
    signature_sha = hmac.new(api_secret.encode("utf-8"),
                             signature_origin.encode("utf-8"),
                             digestmod=hashlib.sha256).digest()
    signature_b64 = base64.b64encode(signature_sha).decode("utf-8")
    authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_b64}"'
    authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")
    q = {"authorization": authorization, "date": date, "host": host}
    qs = (u.query + "&" if u.query else "") + "&".join([f"{k}={urllib.parse.quote(v)}" for k,v in q.items()])
    return urllib.parse.urlunparse((u.scheme, u.netloc, u.path, u.params, qs, u.fragment))

def _ifly_build_payload(app_id: str, text: str, vcn: str, speed: int, volume: int, pitch: int,
                        encoding: str, sample_rate: int, channels: int, bit_depth: int,
                        oral_enabled: bool, oral_level: str, stop_split: int, remain: int):
    header = {"app_id": app_id, "status": 2}
    tts = {
        "vcn": vcn, "speed": int(speed), "volume": int(volume), "pitch": int(pitch),
        "bgs": 0, "reg": 0, "rdn": 0, "rhy": 0,
        "audio": {
            "encoding": encoding, "sample_rate": int(sample_rate),
            "channels": int(channels), "bit_depth": int(bit_depth), "frame_size": 0
        }
    }
    parameter = {"tts": tts}
    if oral_enabled:
        parameter["oral"] = {
            "oral_level": oral_level, "spark_assist": 1,
            "stop_split": int(stop_split), "remain": int(remain)
        }
    payload = {
        "text": {
            "encoding": "utf8", "compress": "raw", "format": "plain",
            "status": 2, "seq": 0,
            "text": base64.b64encode(text.encode("utf-8")).decode("utf-8")
        }
    }
    return {"header": header, "parameter": parameter, "payload": payload}

def iflytek_ws_tts(text: str, ws_url: str, app_id: str, api_key: str, api_secret: str,
                   vcn: str="x5_lingfeiyi_flow", speed: int=50, volume: int=50, pitch: int=50,
                   encoding: str="lame", sample_rate: int=24000, channels: int=1, bit_depth: int=16,
                   oral_level: str="mid", stop_split: int=0, remain: int=0) -> bytes:
    if not text.strip():
        return b""
    oral_enabled = not vcn.startswith("x5_")   # x5 系列不支持 oral
    signed = _ifly_sign_ws_url(ws_url, api_key, api_secret)
    from websocket import create_connection
    ws = create_connection(signed, sslopt={"cert_reqs": ssl.CERT_NONE})
    try:
        req = _ifly_build_payload(app_id, text, vcn, speed, volume, pitch,
                                  encoding, sample_rate, channels, bit_depth,
                                  oral_enabled, oral_level, stop_split, remain)
        ws.send(json.dumps(req, ensure_ascii=False))
        audio_buf = bytearray(); sid=None
        while True:
            msg = ws.recv()
            if not msg: break
            resp = json.loads(msg)
            h = resp.get("header", {})
            code = h.get("code", 0); sid = h.get("sid") or sid
            if code != 0:
                raise RuntimeError(f"iflytek error: code={code} msg={h.get('message')} sid={sid}")
            payload = resp.get("payload", {})
            audio = payload.get("audio") or {}
            a_b64 = audio.get("audio")
            if a_b64:
                audio_buf.extend(base64.b64decode(a_b64))
            if h.get("status") == 2: break
        return bytes(audio_buf)
    finally:
        try: ws.close()
        except: pass

def save_audio_iflytek_from_sentences(sentences: List[str],
                                      ws_url: str, app_id: str, api_key: str, api_secret: str,
                                      vcn: str, speed: int, volume: int, pitch: int,
                                      encoding: str, sample_rate: int, channels: int, bit_depth: int,
                                      oral_level: str, stop_split: int, remain: int,
                                      out_path: Path, on_empty: str, by_sent: bool, pause_ms: int):
    sents = [s.strip() for s in (sentences or []) if s.strip()]
    if not sents:
        if on_empty=="silence":
            AudioSegment.silent(duration=600).export(out_path, format="mp3"); return
        sents = ["这一页没有可读文本"]
    if by_sent:
        # 用官方 [pXXX] 停顿标记，合成一次即可
        text = f"[p{int(pause_ms)}]".join(sents)
    else:
        text = join_for_tts(sents, sep=' ')
    audio = iflytek_ws_tts(
        text=text, ws_url=ws_url, app_id=app_id, api_key=api_key, api_secret=api_secret,
        vcn=vcn, speed=speed, volume=volume, pitch=pitch,
        encoding=encoding, sample_rate=sample_rate, channels=channels, bit_depth=bit_depth,
        oral_level=oral_level, stop_split=stop_split, remain=remain
    )
    Path(out_path).write_bytes(audio)

# ---------------- 并行 worker：OCR+TTS ----------------
def worker_process_one(img_path_str: str,
                       page_index: Optional[int],
                       ocr_engine: str, lang_ocr: str, ocr_psm: int, ocr_oem: int,
                       ocr_crop: Optional[Tuple[float,float,float,float]],
                       strip_border: float, ocr_scale: float, binarize: bool,
                       tencent_model: str,
                       tts_engine: str,
                       # edge
                       edge_voice: str, edge_rate: str, edge_pitch: str, edge_style: Optional[str],
                       # azure
                       azure_voice: str, azure_rate: str, azure_pitch: str, azure_style: Optional[str],
                       # elevenlabs
                       eleven_voice_id: Optional[str],
                       # gtts/pyttsx3
                       voice_lang: str, voice_name: Optional[str],
                       # chatts
                       chatts_url: Optional[str], chatts_prompt: Optional[str], chatts_speaker: Optional[str],
                       chatts_seed: Optional[int], chatts_lang: Optional[str],
                       chatts_refine: bool, chatts_refine_prompt: Optional[str], chatts_refine_seed: Optional[int],
                       chatts_normalize: bool, chatts_homophone: bool, chatts_timeout: float,
                       # indextts
                       indextts_url: Optional[str], indextts_prompt_wav: Optional[str],
                       indextts_emo_wav: Optional[str],
                       indextts_emo_text: Optional[str], indextts_use_emo_text: bool,
                       indextts_emo_alpha: float, indextts_emo_vector_json: Optional[str],
                       indextts_use_random: bool, indextts_interval_silence: Optional[int],
                       indextts_max_text_tokens: Optional[int], indextts_temperature: Optional[float],
                       indextts_top_p: Optional[float], indextts_top_k: Optional[int],
                       indextts_repetition_penalty: Optional[float], indextts_max_mel_tokens: Optional[int],
                       indextts_timeout: float,
                       # iflytek
                       ifly_ws_url: Optional[str], ifly_app_id: Optional[str], ifly_key: Optional[str], ifly_secret: Optional[str],
                       ifly_vcn: str, ifly_speed: int, ifly_volume: int, ifly_pitch: int,
                       ifly_encoding: str, ifly_sr: int, ifly_channels: int, ifly_bit_depth: int,
                       ifly_oral_level: str, ifly_stop_split: int, ifly_remain: int,
                       ifly_by_sent: bool, ifly_pause_ms: int,
                       audio_dir_str: str, text_dir_str: str,
                       on_empty: str, pad_silence: float, debug: bool, force_ocr: bool,
                       filter_keywords: Tuple[str, ...]):
    img_path = Path(img_path_str)
    audio_dir, text_dir = Path(audio_dir_str), Path(text_dir_str)

    logs_dir = audio_dir.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = logs_dir / f"{img_path.stem}.ocr.{ocr_engine}.touch"

    text_path = text_dir / f"{img_path.stem}.txt"
    audio_path = audio_dir / f"{img_path.stem}.mp3"

    stem = img_path.stem
    skip_set = set([s.strip() for s in (os.getenv("__SKIP_STEMS_RUNTIME","").split(",")) if s.strip()])
    if stem in skip_set:
        text_path.write_text("", encoding="utf-8")
        dur_ms = int(float(os.getenv("__SKIP_SIL_MS_RUNTIME","1000")))
        AudioSegment.silent(duration=dur_ms).export(audio_path, format="mp3")
        if debug: print(f"[SKIP] {stem} -> silent {dur_ms/1000.0:.2f}s")
        try:
            with open(logs_dir / f"{img_path.stem}.ocr.skipped.touch", "w", encoding="utf-8") as f:
                f.write("engine=skipped\nlen=0\n")
        except: pass
        dur = AudioSegment.from_file(audio_path).duration_seconds
        if debug: print(f"[AUDIO] {img_path.name} -> {dur:.2f}s")
        return (str(img_path), str(text_path), str(audio_path), float(dur))

    # OCR
    need_ocr = force_ocr or (not text_path.exists()) or (text_path.read_text(encoding="utf-8", errors="ignore").strip() == "")
    if need_ocr:
        try:
            raw_text = ocr_image(img_path, ocr_engine, lang_ocr, ocr_psm, ocr_oem,
                                 ocr_crop, strip_border, ocr_scale, binarize,
                                 tencent_model, debug, page_index=page_index)
            text = clean_text_for_manhua(raw_text)
            if filter_keywords:
                text = filter_text_by_keywords(text, filter_keywords)
        except Exception as e:
            print(f"[OCR-ERR] {ocr_engine} {img_path.name}: {e}", file=sys.stderr)
            raise
        text_path.write_text(text, encoding="utf-8")
    else:
        text = text_path.read_text(encoding="utf-8", errors="ignore")
        if debug: print(f"[OCR] cache HIT -> {img_path.name} (chars={len(text)})")
        if filter_keywords:
            filtered = filter_text_by_keywords(text, filter_keywords)
            if filtered != text:
                text = filtered
                text_path.write_text(text, encoding="utf-8")

    try:
        with open(stamp_path, "w", encoding="utf-8") as f:
            f.write(f"engine={ocr_engine}\nmodel={tencent_model if ocr_engine=='tencent' else '-'}\nlen={len(text)}\n")
    except: pass

    paragraph = " ".join(line.strip() for line in text.splitlines() if line.strip())
    sentences = [paragraph] if paragraph else []

    # TTS
    if not audio_path.exists():
        try:
            if tts_engine == "edge":
                save_audio_edge_tts_from_sentences(sentences, edge_voice, edge_rate, None, None, audio_path, on_empty)
            elif tts_engine == "azure":
                save_audio_azure_tts_from_sentences(sentences, azure_voice, azure_rate, azure_pitch, azure_style, audio_path, on_empty)
            elif tts_engine == "elevenlabs":
                v = eleven_voice_id or os.getenv("ELEVEN_VOICE_ID","")
                if not v: raise RuntimeError("缺少 ELEVEN_VOICE_ID（.env或参数）")
                save_audio_elevenlabs(join_for_tts(sentences, ' '), v, audio_path, on_empty)
            elif tts_engine == "gtts":
                save_audio_gtts_from_sentences(sentences, voice_lang, audio_path, on_empty)
            elif tts_engine == "pyttsx3":
                save_audio_pyttsx3_from_sentences(sentences, voice_name, audio_path, on_empty)
            elif tts_engine == "chatts":
                api_url = (chatts_url or "http://127.0.0.1:9900/generate_voice").strip()
                if not api_url:
                    raise RuntimeError("缺少 ChatTTS API 地址（CHATTTS_API_URL 或 --chatts-url）")
                save_audio_chatts_from_sentences(
                    sentences,
                    api_url,
                    chatts_prompt or "[speed_5]",
                    chatts_speaker,
                    chatts_seed,
                    chatts_lang,
                    chatts_refine,
                    chatts_refine_prompt,
                    chatts_refine_seed,
                    chatts_normalize,
                    chatts_homophone,
                    chatts_timeout,
                    audio_path,
                    on_empty,
                )
            elif tts_engine == "indextts":
                api_url = (indextts_url or "http://127.0.0.1:8000/tts").strip()
                save_audio_indextts_from_sentences(
                    sentences,
                    api_url,
                    indextts_prompt_wav or "",
                    indextts_emo_wav,
                    indextts_emo_text,
                    bool(indextts_use_emo_text),
                    float(indextts_emo_alpha),
                    indextts_emo_vector_json,
                    bool(indextts_use_random),
                    indextts_interval_silence,
                    indextts_max_text_tokens,
                    indextts_temperature,
                    indextts_top_p,
                    indextts_top_k,
                    indextts_repetition_penalty,
                    indextts_max_mel_tokens,
                    indextts_timeout,
                    audio_path,
                    on_empty,
                )
            elif tts_engine == "http":
                api_url = os.getenv("TTS_API_URL", "http://127.0.0.1:8002/tts")
                tts_speed = float(os.getenv("TTS_SPEED", "1.0"))
                save_audio_http_tts_from_sentences(sentences, api_url, voice_lang, voice_name, tts_speed, audio_path, on_empty)
            elif tts_engine == "iflytek":
                ws_url = ifly_ws_url or os.getenv("XFYUN_WS_URL")
                app_id = ifly_app_id or os.getenv("XFYUN_APPID")
                api_key = ifly_key or os.getenv("XFYUN_API_KEY")
                api_secret = ifly_secret or os.getenv("XFYUN_API_SECRET")
                if not (ws_url and app_id and api_key and api_secret):
                    raise RuntimeError("讯飞私有域缺少配置：XFYUN_WS_URL / XFYUN_APPID / XFYUN_API_KEY / XFYUN_API_SECRET")
                save_audio_iflytek_from_sentences(
                    sentences, ws_url, app_id, api_key, api_secret,
                    ifly_vcn, ifly_speed, ifly_volume, ifly_pitch,
                    ifly_encoding, ifly_sr, ifly_channels, ifly_bit_depth,
                    ifly_oral_level, ifly_stop_split, ifly_remain,
                    audio_path, on_empty, ifly_by_sent, ifly_pause_ms
                )
            else:
                raise ValueError(f"未知 tts-engine: {tts_engine}")
        except Exception as e:
            print(f"[TTS-ERR] {tts_engine} {img_path.name}: {e}", file=sys.stderr)
            raise

        if pad_silence and pad_silence>0.01:
            seg = AudioSegment.from_file(audio_path)
            seg += AudioSegment.silent(duration=int(pad_silence*1000))
            seg.export(audio_path, format="mp3")

    dur = AudioSegment.from_file(audio_path).duration_seconds
    if debug: print(f"[AUDIO] {img_path.name} -> {dur:.2f}s")
    return (str(img_path), str(text_path), str(audio_path), float(dur))

# ---------------- 分段编码 + 快速拼接 ----------------
def _encode_one_segment_ffmpeg(img_path: Path, audio_path: Path, seg_out: Path,
                               width: int, height: int, fps: int, fade: float,
                               preset: str, crf: int, threads: int, audio_sr=44100):
    dur = max(0.1, AudioSegment.from_file(audio_path).duration_seconds)
    vf = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "format=yuv420p"
    ]
    if fade and fade>0:
        st_out = max(0.0, dur - fade)
        vf += [f"fade=t=in:st=0:d={fade}", f"fade=t=out:st={st_out}:d={fade}"]
    vf_str = ",".join(vf)
    cmd = [
        "ffmpeg","-y",
        "-loop","1","-framerate",str(fps),"-i",str(img_path),
        "-i",str(audio_path),
        "-t",f"{dur:.3f}",
        "-vf",vf_str,
        "-r",str(fps),
        "-c:v","libx264","-preset",preset,"-crf",str(crf),
        "-c:a","aac","-b:a","128k","-ar",str(audio_sr),"-ac","2",
        "-shortest",
        "-threads",str(threads),
        "-movflags","+faststart",
        str(seg_out)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def fast_concat_segments(seg_paths: List[Path], out_path: Path):
    lst = out_path.parent / "concat_list.txt"
    with open(lst,"w",encoding="utf-8") as f:
        for p in seg_paths:
            f.write(f"file '{p.as_posix()}'\n")
    cmd = ["ffmpeg","-y","-f","concat","-safe","0","-i",str(lst),"-c","copy","-movflags","+faststart",str(out_path)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

# ---------------- MoviePy 常规管线（需要字幕时） ----------------
def make_clip_with_audio(image_path: Path, audio_path: Path, subtitle_text: Optional[str],
                         target_res: Optional[Tuple[int,int]], fade: float, fps: int) -> CompositeVideoClip:
    img = Image.open(image_path).convert("RGB")
    if subtitle_text:
        W,H = img.size
        font = ensure_font(max(18, H//36))
        pad = max(12, H//100)
        box_w, box_h = int(W*0.9), int(H*0.30)
        x = (W - box_w)//2; y = H - box_h - int(H*0.05)
        overlay = Image.new("RGBA", img.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        try: draw.rounded_rectangle([x,y,x+box_w,y+box_h], radius=16, fill=(0,0,0,140))
        except: draw.rectangle([x,y,x+box_w,y+box_h], fill=(0,0,0,140))
        def wrap(text, maxw):
            lines=[]; buf=""
            for ch in text:
                trial=buf+ch
                if font.getsize(trial)[0]<=maxw: buf=trial
                else: lines.append(buf); buf=ch
            if buf: lines.append(buf)
            return lines
        lines = wrap(subtitle_text, box_w-2*pad)
        line_h = font.getsize("字")[1]+4
        total_h = len(lines)*line_h; cy = y + (box_h-total_h)//2
        for line in lines:
            w,_ = font.getsize(line)
            draw.text((x+(box_w-w)//2, cy), line, font=font, fill=(255,255,255,255)); cy+=line_h
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    if target_res and target_res[0]>0 and target_res[1]>0:
        img = img.resize(target_res)
    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp_img.name); img.close()
    audio = AudioFileClip(str(audio_path))
    duration = max(0.1, audio.duration)
    clip = ImageClip(tmp_img.name).with_duration(duration).with_audio(audio).with_fps(fps)
    if fade and fade>0:
        clip = clip.with_effects([vfx.FadeIn(fade), vfx.FadeOut(fade)])
    comp = CompositeVideoClip([clip]).with_duration(duration)
    comp._tmp_image_path = tmp_img.name
    return comp

# ---------------- 主流程 ----------------
def default_workers(tts_engine: str) -> int:
    n = os.cpu_count() or 4
    return 2 if tts_engine in ("edge","azure","gtts","elevenlabs","iflytek","chatts","indextts") else n

def build_video(images: List[Path], out_path: Path,
                ocr_engine: str, lang_ocr: str, ocr_psm: int, ocr_oem: int,
                ocr_crop: Optional[Tuple[float,float,float,float]],
                strip_border: float, ocr_scale: float, binarize: bool,
                tencent_model: str,
                tts_engine: str,
                edge_voice: str, edge_rate: str, edge_pitch: str, edge_style: Optional[str],
                azure_voice: str, azure_rate: str, azure_pitch: str, azure_style: Optional[str],
                eleven_voice_id: Optional[str], voice_lang: str, voice_name: Optional[str],
                chatts_url: Optional[str], chatts_prompt: Optional[str], chatts_speaker: Optional[str],
                chatts_seed: Optional[int], chatts_lang: Optional[str],
                chatts_refine: bool, chatts_refine_prompt: Optional[str], chatts_refine_seed: Optional[int],
                chatts_normalize: bool, chatts_homophone: bool, chatts_timeout: float,
                indextts_url: Optional[str], indextts_prompt_wav: Optional[str],
                indextts_emo_wav: Optional[str], indextts_emo_text: Optional[str],
                indextts_use_emo_text: bool, indextts_emo_alpha: float,
                indextts_emo_vector_json: Optional[str], indextts_use_random: bool,
                indextts_interval_silence: Optional[int], indextts_max_text_tokens: Optional[int],
                indextts_temperature: Optional[float], indextts_top_p: Optional[float],
                indextts_top_k: Optional[int], indextts_repetition_penalty: Optional[float],
                indextts_max_mel_tokens: Optional[int], indextts_timeout: float,
                # iflytek
                ifly_ws_url: Optional[str], ifly_app_id: Optional[str], ifly_key: Optional[str], ifly_secret: Optional[str],
                ifly_vcn: str, ifly_speed: int, ifly_volume: int, ifly_pitch: int,
                ifly_encoding: str, ifly_sr: int, ifly_channels: int, ifly_bit_depth: int,
                ifly_oral_level: str, ifly_stop_split: int, ifly_remain: int,
                ifly_by_sent: bool, ifly_pause_ms: int,
                subtitle: bool, fade: float, fps: int,
                target_width: Optional[int], target_height: Optional[int],
                pad_silence: float, on_empty: str, workers: int,
                fast_concat: bool, enc_workers: int, enc_threads: int,
                enc_preset: str, enc_crf: int,
                debug: bool, force_ocr: bool,
                skip_stems: str, skip_silence_sec: float,
                filter_keywords: Tuple[str, ...]):
    work = out_path.parent / "_ctv_build"
    audio_dir, text_dir, seg_dir = work/"audio", work/"text", work/"seg"
    for d in (work, audio_dir, text_dir, seg_dir): d.mkdir(parents=True, exist_ok=True)

    os.environ["__SKIP_STEMS_RUNTIME"] = skip_stems or ""
    os.environ["__SKIP_SIL_MS_RUNTIME"] = str(int(max(0.0, skip_silence_sec) * 1000))

    if workers<=0: workers = default_workers(tts_engine)

    print(f"[BOOT] workers={workers} fast_concat={fast_concat} enc_workers={enc_workers} enc_threads={enc_threads}")
    results: Dict[str, Tuple[str,str,float]] = {}

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs=[]
        for page_index, img in enumerate(images, start=1):
            futs.append(ex.submit(
                worker_process_one, str(img), page_index,
                ocr_engine, lang_ocr, ocr_psm, ocr_oem,
                ocr_crop, strip_border, ocr_scale, binarize,
                tencent_model,
                tts_engine,
                edge_voice, edge_rate, edge_pitch, edge_style,
                azure_voice, azure_rate, azure_pitch, azure_style,
                eleven_voice_id,
                voice_lang, voice_name,
                chatts_url, chatts_prompt, chatts_speaker,
                chatts_seed, chatts_lang,
                chatts_refine, chatts_refine_prompt, chatts_refine_seed,
                chatts_normalize, chatts_homophone, chatts_timeout,
                indextts_url, indextts_prompt_wav,
                indextts_emo_wav, indextts_emo_text,
                indextts_use_emo_text, indextts_emo_alpha,
                indextts_emo_vector_json, indextts_use_random,
                indextts_interval_silence, indextts_max_text_tokens,
                indextts_temperature, indextts_top_p,
                indextts_top_k, indextts_repetition_penalty,
                indextts_max_mel_tokens, indextts_timeout,
                ifly_ws_url, ifly_app_id, ifly_key, ifly_secret,
                ifly_vcn, ifly_speed, ifly_volume, ifly_pitch,
                ifly_encoding, ifly_sr, ifly_channels, ifly_bit_depth,
                ifly_oral_level, ifly_stop_split, ifly_remain,
                ifly_by_sent, ifly_pause_ms,
                str(audio_dir), str(text_dir),
                on_empty, pad_silence, debug, force_ocr,
                filter_keywords
            ))
        for f in as_completed(futs):
            img_path, text_path, audio_path, dur = f.result()
            results[img_path] = (text_path, audio_path, dur)

    if fast_concat and not subtitle:
        print("[PIPE] 快速分段编码 + 0重编码拼接")
        if not (target_width and target_height):
            w,h = Image.open(images[0]).size
            target_width, target_height = int(w), int(h)
        if enc_workers<=0: enc_workers = os.cpu_count() or 4
        seg_paths=[]
        with ProcessPoolExecutor(max_workers=enc_workers) as ex:
            futs=[]
            for i,img in enumerate(images,1):
                _, audio_path, _ = results[str(img)]
                seg_out = seg_dir / f"{i:04d}.mp4"
                seg_paths.append(seg_out)
                futs.append(ex.submit(
                    _encode_one_segment_ffmpeg, img, Path(audio_path), seg_out,
                    int(target_width), int(target_height), int(fps), float(fade),
                    enc_preset, int(enc_crf), int(enc_threads)
                ))
            for _ in as_completed(futs): pass
        fast_concat_segments(seg_paths, out_path)
        print("[DONE]", out_path)
    else:
        print("[PIPE] MoviePy 常规管线（字幕/禁用 fast-concat）")
        target_res = (int(target_width), int(target_height)) if (target_width and target_height) else None
        clips, tmp_imgs = [], []
        for i,img in enumerate(images,1):
            text_path, audio_path, _ = results[str(img)]
            txt = Path(text_path).read_text(encoding="utf-8", errors="ignore")
            print(f"[CLIP] {i}/{len(images)} {img.name}")
            clip = make_clip_with_audio(img, Path(audio_path), txt if subtitle else None,
                                        target_res, fade, fps)
            clips.append(clip); tmp_imgs.append(getattr(clip,"_tmp_image_path",None))
        ffargs = ["-preset","veryfast","-crf","23","-threads", str(max(1, os.cpu_count() or 1))]
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(str(out_path), fps=fps, codec="libx264", audio_codec="aac", ffmpeg_params=ffargs)
        for p in tmp_imgs:
            try:
                if p and os.path.exists(p): os.remove(p)
            except: pass

    # 汇总
    log_dir = out_path.parent / "_ctv_build" / "logs"
    counts = {}
    if log_dir.exists():
        for p in log_dir.iterdir():
            if p.name.endswith(".touch"):
                sfx = p.suffixes
                eng = sfx[-2].lstrip(".") if len(sfx)>=2 else "unknown"
                counts[eng] = counts.get(eng, 0) + 1
    print("[OCR-SUMMARY]", counts)

def pick_engine(cli_value: str, env_key: str, default_value: str, allowed: List[str]) -> str:
    if cli_value and cli_value.lower() != "auto":
        v = cli_value.lower()
        if v not in allowed: raise ValueError(f"{env_key} 取值不支持：{v}")
        return v
    ev = os.getenv(env_key, default_value).lower()
    return ev if ev in allowed else default_value

def main():
    ap = argparse.ArgumentParser(description="连环画 -> 云OCR/本地OCR + 多TTS -> 极速视频（含 iFLYTEK 私有域）")
    ap.add_argument("--env-file", default=".env", help="指定 .env 路径")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--force-ocr", action="store_true")

    # OCR
    ap.add_argument("--ocr-engine", choices=["auto","tencent","baidu","rapidocr","tesseract","http"], default="auto")
    ap.add_argument("--tencent-model", choices=["basic","accurate"], default=None)
    ap.add_argument("--lang", default="chi_sim+eng")   # tesseract
    ap.add_argument("--ocr-psm", type=int, default=6)
    ap.add_argument("--ocr-oem", type=int, default=1)
    ap.add_argument("--ocr-crop", type=str, default=None, help="相对裁剪 x1,y1,x2,y2（0~1）")
    ap.add_argument("--strip-border", type=float, default=0.01)
    ap.add_argument("--scale", type=float, default=1.7)
    ap.add_argument("--no-binarize", action="store_true")

    # TTS 总开关
    ap.add_argument("--tts-engine", choices=["auto","edge","azure","elevenlabs","gtts","pyttsx3","iflytek","http","chatts","indextts"], default="auto")

    # edge（pitch/style 忽略以兼容老版 edge-tts）
    ap.add_argument("--edge-voice", default=None)
    ap.add_argument("--edge-rate", default=None)
    ap.add_argument("--edge-pitch", default=None, help="已兼容忽略")
    ap.add_argument("--edge-style", default=None, help="已兼容忽略")

    # azure
    ap.add_argument("--azure-voice", default=None)
    ap.add_argument("--azure-rate", default=None)
    ap.add_argument("--azure-pitch", default=None)
    ap.add_argument("--azure-style", default=None)

    # elevenlabs
    ap.add_argument("--eleven-voice-id", default=None)

    # gtts/pyttsx3
    ap.add_argument("--voice-lang", default=None)
    ap.add_argument("--voice-name", default=None)

    # chatts
    ap.add_argument("--chatts-url", default=None)
    ap.add_argument("--chatts-prompt", default=None)
    ap.add_argument("--chatts-speaker", default=None)
    ap.add_argument("--chatts-seed", type=int, default=None)
    ap.add_argument("--chatts-lang", default=None)
    ap.add_argument("--chatts-refine", action="store_true")
    ap.add_argument("--chatts-refine-prompt", default=None)
    ap.add_argument("--chatts-refine-seed", type=int, default=None)
    ap.add_argument("--chatts-no-normalize", action="store_true")
    ap.add_argument("--chatts-homophone", action="store_true")
    ap.add_argument("--chatts-timeout", type=float, default=None)

    # iflytek（私有域）
    ap.add_argument("--ifly-ws-url", default=None, help="例：wss://cbm01.cn-huabei-1.xf-yun.com/v1/private/xxxx")
    ap.add_argument("--ifly-app-id", default=None)
    ap.add_argument("--ifly-api-key", default=None)
    ap.add_argument("--ifly-api-secret", default=None)
    ap.add_argument("--ifly-vcn", default="x5_lingfeiyi_flow")
    ap.add_argument("--ifly-speed", type=int, default=50)
    ap.add_argument("--ifly-volume", type=int, default=50)
    ap.add_argument("--ifly-pitch", type=int, default=50)
    ap.add_argument("--ifly-encoding", choices=["lame","raw"], default="lame")
    ap.add_argument("--ifly-sr", type=int, default=24000)
    ap.add_argument("--ifly-channels", type=int, default=1)
    ap.add_argument("--ifly-bit-depth", type=int, default=16)
    ap.add_argument("--ifly-oral-level", choices=["high","mid","low"], default="mid")
    ap.add_argument("--ifly-stop-split", type=int, default=0)
    ap.add_argument("--ifly-remain", type=int, default=0)
    ap.add_argument("--ifly-by-sent", action="store_true", help="按句切分，用 [pXXX] 停顿拼成一次请求")
    ap.add_argument("--ifly-pause-ms", type=int, default=350)

    # 行为
    ap.add_argument("--on-empty", choices=["speak","silence"], default="silence")
    ap.add_argument("--pad-silence", type=float, default=0.2)

    # 视频与并行
    ap.add_argument("--subtitle", choices=["on","off"], default="off")
    ap.add_argument("--fade", type=float, default=0.25)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--target-width", type=int, default=None)
    ap.add_argument("--target-height", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)

    # 极速拼接
    ap.add_argument("--fast-concat", choices=["on","off"], default="on")
    ap.add_argument("--enc-workers", type=int, default=4)
    ap.add_argument("--enc-threads", type=int, default=1)
    ap.add_argument("--enc-preset", default="veryfast")
    ap.add_argument("--enc-crf", type=int, default=23)

    # 跳过页
    default_skip_stems = ""
    default_skip_silence = 1.0
    default_filter_keywords = ""

    ap.add_argument("--skip-stems", default=default_skip_stems, help="例：01,01p")
    ap.add_argument("--skip-silence-sec", type=float, default=default_skip_silence)
    ap.add_argument("--filter-keywords", default=default_filter_keywords, help="逗号分隔，出现则整行丢弃")

    args = ap.parse_args()
    load_dotenv(args.env_file)

    env_file_path = Path(args.env_file).expanduser().resolve()
    env_base_dir = env_file_path.parent

    def resolve_env_path(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = value.strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (env_base_dir / path).resolve()
        return str(path)

    input_path = Path(args.images_dir).expanduser().resolve()
    if not input_path.exists():
        print(f"输入路径不存在：{input_path}", file=sys.stderr)
        sys.exit(1)
    out_path = Path(args.out).expanduser().resolve()
    pdf_image_root = out_path.parent / "_ctv_build" / "pdf_images"

    images: List[Path] = []

    def _convert_and_collect(pdf_file: Path) -> List[Path]:
        print(f"[PDF] 开始转换：{pdf_file}")
        try:
            converted = convert_pdf_to_images(pdf_file, pdf_image_root / pdf_file.stem)
        except ImportError:
            print("缺少 PyMuPDF，请先安装：pip install PyMuPDF", file=sys.stderr)
            sys.exit(1)
        except Exception as err:
            print(f"PDF 转图片失败：{pdf_file} -> {err}", file=sys.stderr)
            sys.exit(1)
        print(f"[PDF] 转换完成：{pdf_file.name} -> {len(converted)} 张图片，输出目录：{pdf_image_root / pdf_file.stem}")
        return converted

    if input_path.is_dir():
        pdfs = list_pdfs_sorted(input_path)
        image_files = list_images_sorted(input_path)
        if pdfs and image_files:
            print(f"目录中同时包含 PDF 和 图片，请分开存放：{input_path}", file=sys.stderr)
            sys.exit(1)
        if pdfs:
            print(f"[INPUT] 检测到 {len(pdfs)} 个 PDF 文件，准备转换为图片...")
            for pdf_file in pdfs:
                images.extend(_convert_and_collect(pdf_file))
        else:
            images = image_files
            print(f"[INPUT] 检测到 {len(images)} 张图片：{input_path}")
    else:
        suffix = input_path.suffix.lower()
        if suffix == ".pdf":
            print("[INPUT] 检测到 PDF 文件，准备转换为图片...")
            images = _convert_and_collect(input_path)
        elif suffix in IMAGE_EXTS:
            images = [input_path]
            print(f"[INPUT] 使用单张图片：{input_path}")
        else:
            print(f"不支持的输入文件类型：{input_path}", file=sys.stderr)
            sys.exit(1)

    if not images:
        print(f"未找到可处理的图片：{input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INPUT] 总计 {len(images)} 张图片待处理。")

    # 选择 OCR / TTS
    ocr_engine = pick_engine(args.ocr_engine, "OCR_VENDOR", "tencent",
                             ["tencent","baidu","rapidocr","tesseract","http"])
    tts_engine = pick_engine(args.tts_engine, "TTS_VENDOR", "edge",
                             ["edge","azure","elevenlabs","gtts","pyttsx3","iflytek","http","chatts","indextts"])
    tencent_model = (args.tencent_model or os.getenv("TENCENT_OCR_MODEL","basic")).lower()
    if tencent_model not in ("basic","accurate"): tencent_model = "basic"

    chatts_url = args.chatts_url or os.getenv("CHATTTS_API_URL")
    chatts_url = chatts_url.strip() if chatts_url else "http://127.0.0.1:9900/generate_voice"
    chatts_prompt = args.chatts_prompt or os.getenv("CHATTTS_PROMPT", "[speed_5]")
    chatts_speaker = args.chatts_speaker or os.getenv("CHATTTS_SPK_EMB")
    chatts_seed = args.chatts_seed if args.chatts_seed is not None else _maybe_int(os.getenv("CHATTTS_SEED"))
    chatts_lang = args.chatts_lang or os.getenv("CHATTTS_LANG")
    chatts_refine = args.chatts_refine or _bool_from(os.getenv("CHATTTS_REFINE"), False)
    chatts_refine_prompt = args.chatts_refine_prompt or os.getenv("CHATTTS_REFINE_PROMPT")
    chatts_refine_seed = args.chatts_refine_seed if args.chatts_refine_seed is not None else _maybe_int(os.getenv("CHATTTS_REFINE_SEED"))
    chatts_normalize = _bool_from(os.getenv("CHATTTS_NORMALIZE"), not args.chatts_no_normalize)
    chatts_homophone = _bool_from(os.getenv("CHATTTS_HOMOPHONE"), args.chatts_homophone)
    if args.chatts_timeout is not None:
        chatts_timeout = args.chatts_timeout
    else:
        timeout_env = os.getenv("CHATTTS_TIMEOUT")
        try:
            chatts_timeout = float(timeout_env) if timeout_env is not None else 120.0
        except (TypeError, ValueError):
            chatts_timeout = 120.0

    def ensure_env_vars(keys: List[str], context: str):
        missing = [k for k in keys if not os.getenv(k)]
        if missing:
            msg = f"缺少 {context} 所需的环境变量：{', '.join(missing)}（请在 .env 或环境中配置）"
            print(msg, file=sys.stderr)
            sys.exit(1)

    print(">>> .env file:", os.path.abspath(args.env_file))
    ocr_msg = f">>> OCR engine chosen: {ocr_engine}"
    if ocr_engine == "tencent":
        ocr_msg += f" (tencent_model={tencent_model})"
        ensure_env_vars(["TENCENT_SECRET_ID", "TENCENT_SECRET_KEY"], "腾讯 OCR")
    elif ocr_engine == "baidu":
        ensure_env_vars(["BAIDU_OCR_AK", "BAIDU_OCR_SK"], "百度 OCR")
    print(ocr_msg)

    print(">>> TTS engine chosen:", tts_engine)
    if tts_engine == "azure":
        ensure_env_vars(["AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION"], "Azure 语音合成")
    elif tts_engine == "elevenlabs":
        ensure_env_vars(["ELEVEN_API_KEY"], "ElevenLabs 语音合成")
    elif tts_engine == "chatts" and not chatts_url:
        print("缺少 ChatTTS API URL（请通过 --chatts-url 或 CHATTTS_API_URL 指定）", file=sys.stderr)
        sys.exit(1)

    # OCR 裁剪
    ocr_crop = None
    if args.ocr_crop:
        try:
            x1,y1,x2,y2 = [float(s) for s in args.ocr_crop.split(",")]
            ocr_crop = (x1,y1,x2,y2)
        except:
            print("无效的 --ocr-crop，格式 x1,y1,x2,y2（0~1）", file=sys.stderr); sys.exit(1)

    # TTS 参数（Edge pitch/style 忽略）
    edge_voice = args.edge_voice or os.getenv("EDGE_VOICE", "zh-CN-XiaoxiaoNeural")
    edge_rate  = args.edge_rate  or os.getenv("EDGE_RATE", "+0%")
    edge_pitch = args.edge_pitch or os.getenv("EDGE_PITCH", None)
    edge_style = args.edge_style or os.getenv("EDGE_STYLE", None)

    azure_voice = args.azure_voice or os.getenv("AZURE_VOICE", "zh-CN-XiaoxiaoNeural")
    azure_rate  = args.azure_rate  or os.getenv("AZURE_RATE", "+0%")
    azure_pitch = args.azure_pitch or os.getenv("AZURE_PITCH", "0%")
    azure_style = args.azure_style or os.getenv("AZURE_STYLE", None)

    eleven_voice_id = args.eleven_voice_id or os.getenv("ELEVEN_VOICE_ID", None)
    if tts_engine == "elevenlabs" and not (eleven_voice_id and eleven_voice_id.strip()):
        print("缺少 ElevenLabs Voice ID（请通过 --eleven-voice-id 或 ELEVEN_VOICE_ID 提供）", file=sys.stderr)
        sys.exit(1)
    voice_lang = args.voice_lang or os.getenv("GTTS_LANG", "zh-CN")
    voice_name = args.voice_name or os.getenv("PYTTSX3_VOICE", None)

    indextts_emotion = (os.getenv("INDEXTTS_EMOTION") or "neutral").strip().lower()
    if not indextts_emotion:
        indextts_emotion = "neutral"
    indextts_url = os.getenv("INDEXTTS_API_URL", "http://127.0.0.1:8000/tts")
    indextts_prompt_wav = resolve_env_path(os.getenv("INDEXTTS_PROMPT_WAV"))
    indextts_emo_wav = resolve_env_path(_env_for_emotion("INDEXTTS_EMO_WAV", indextts_emotion))
    indextts_emo_text_raw = _env_for_emotion("INDEXTTS_EMO_TEXT", indextts_emotion)
    indextts_emo_text = indextts_emo_text_raw.strip() if indextts_emo_text_raw else None
    indextts_use_emo_text = _bool_from(_env_for_emotion("INDEXTTS_USE_EMO_TEXT", indextts_emotion), False)
    emo_alpha_val = _env_for_emotion("INDEXTTS_EMO_ALPHA", indextts_emotion)
    indextts_emo_alpha = _maybe_float(emo_alpha_val) if emo_alpha_val is not None else None
    if indextts_emo_alpha is None:
        indextts_emo_alpha = 1.0
    indextts_emo_vector_json_raw = _env_for_emotion("INDEXTTS_EMO_VECTOR_JSON", indextts_emotion)
    indextts_emo_vector_json = indextts_emo_vector_json_raw.strip() if indextts_emo_vector_json_raw else None
    indextts_use_random = _bool_from(_env_for_emotion("INDEXTTS_USE_RANDOM", indextts_emotion), False)
    interval_env = _env_for_emotion("INDEXTTS_INTERVAL_SILENCE", indextts_emotion)
    indextts_interval_silence = _maybe_int(interval_env)
    if indextts_interval_silence is None:
        indextts_interval_silence = 200
    max_tokens_env = _env_for_emotion("INDEXTTS_MAX_TEXT_TOKENS", indextts_emotion)
    indextts_max_text_tokens = _maybe_int(max_tokens_env)
    temperature_env = _env_for_emotion("INDEXTTS_TEMPERATURE", indextts_emotion)
    indextts_temperature = _maybe_float(temperature_env)
    top_p_env = _env_for_emotion("INDEXTTS_TOP_P", indextts_emotion)
    indextts_top_p = _maybe_float(top_p_env)
    top_k_env = _env_for_emotion("INDEXTTS_TOP_K", indextts_emotion)
    indextts_top_k = _maybe_int(top_k_env)
    rep_env = _env_for_emotion("INDEXTTS_REPETITION_PENALTY", indextts_emotion)
    indextts_repetition_penalty = _maybe_float(rep_env)
    max_mel_env = _env_for_emotion("INDEXTTS_MAX_MEL_TOKENS", indextts_emotion)
    indextts_max_mel_tokens = _maybe_int(max_mel_env)
    timeout_env = _env_for_emotion("INDEXTTS_TIMEOUT", indextts_emotion)
    indextts_timeout = _maybe_float(timeout_env)
    if indextts_timeout is None or indextts_timeout <= 0:
        indextts_timeout = 180.0

    # iflytek 优先取命令行，否则取 .env
    ifly_ws_url = args.ifly_ws_url or os.getenv("XFYUN_WS_URL")
    ifly_app_id = args.ifly_app_id or os.getenv("XFYUN_APPID")
    ifly_key    = args.ifly_api_key or os.getenv("XFYUN_API_KEY")
    ifly_secret = args.ifly_api_secret or os.getenv("XFYUN_API_SECRET")
    if tts_engine == "iflytek":
        missing_cfg = []
        if not (ifly_ws_url and ifly_ws_url.strip()):
            missing_cfg.append("XFYUN_WS_URL 或 --ifly-ws-url")
        if not (ifly_app_id and ifly_app_id.strip()):
            missing_cfg.append("XFYUN_APPID 或 --ifly-app-id")
        if not (ifly_key and ifly_key.strip()):
            missing_cfg.append("XFYUN_API_KEY 或 --ifly-api-key")
        if not (ifly_secret and ifly_secret.strip()):
            missing_cfg.append("XFYUN_API_SECRET 或 --ifly-api-secret")
        if missing_cfg:
            print("讯飞私有域缺少配置：" + ", ".join(missing_cfg), file=sys.stderr)
            sys.exit(1)
    if tts_engine == "indextts":
        if not indextts_prompt_wav:
            print("缺少 IndexTTS 参考音频（请通过 INDEXTTS_PROMPT_WAV 设置）", file=sys.stderr)
            sys.exit(1)
        prompt_path = Path(indextts_prompt_wav)
        if not prompt_path.exists():
            print(f"IndexTTS 参考音频不存在：{prompt_path}", file=sys.stderr)
            sys.exit(1)
        if indextts_emo_wav:
            emo_path = Path(indextts_emo_wav)
            if not emo_path.exists():
                print(f"IndexTTS 情感参考音频不存在：{emo_path}", file=sys.stderr)
                sys.exit(1)

    skip_stems_value = args.skip_stems
    if skip_stems_value == default_skip_stems:
        skip_stems_value = os.getenv("SKIP_STEMS", default_skip_stems)

    skip_silence_value = args.skip_silence_sec
    if skip_silence_value == default_skip_silence:
        skip_silence_env = os.getenv("SKIP_SILENCE_SEC")
        if skip_silence_env is not None and skip_silence_env.strip():
            try:
                skip_silence_value = float(skip_silence_env)
            except ValueError:
                print(f"无效的 SKIP_SILENCE_SEC：{skip_silence_env}，已回退为 {default_skip_silence}", file=sys.stderr)
                skip_silence_value = default_skip_silence

    filter_keywords_value = args.filter_keywords
    if filter_keywords_value == default_filter_keywords:
        filter_keywords_value = os.getenv("FILTER_KEYWORDS", default_filter_keywords)
    filter_keywords_tuple = tuple([kw.strip() for kw in filter_keywords_value.replace("\n", ",").split(",") if kw.strip()])

    build_video(
        images=images,
        out_path=out_path,
        ocr_engine=ocr_engine,
        lang_ocr=args.lang,
        ocr_psm=args.ocr_psm,
        ocr_oem=args.ocr_oem,
        ocr_crop=ocr_crop,
        strip_border=args.strip_border,
        ocr_scale=args.scale,
        binarize=not args.no_binarize,
        tencent_model=tencent_model,
        tts_engine=tts_engine,
        edge_voice=edge_voice, edge_rate=edge_rate, edge_pitch=edge_pitch, edge_style=edge_style,
        azure_voice=azure_voice, azure_rate=azure_rate, azure_pitch=azure_pitch, azure_style=azure_style,
        eleven_voice_id=eleven_voice_id,
        voice_lang=voice_lang, voice_name=voice_name,
        chatts_url=chatts_url, chatts_prompt=chatts_prompt, chatts_speaker=chatts_speaker,
        chatts_seed=chatts_seed, chatts_lang=chatts_lang,
        chatts_refine=chatts_refine, chatts_refine_prompt=chatts_refine_prompt, chatts_refine_seed=chatts_refine_seed,
        chatts_normalize=chatts_normalize, chatts_homophone=chatts_homophone, chatts_timeout=chatts_timeout,
        indextts_url=indextts_url, indextts_prompt_wav=indextts_prompt_wav,
        indextts_emo_wav=indextts_emo_wav, indextts_emo_text=indextts_emo_text,
        indextts_use_emo_text=indextts_use_emo_text, indextts_emo_alpha=indextts_emo_alpha,
        indextts_emo_vector_json=indextts_emo_vector_json, indextts_use_random=indextts_use_random,
        indextts_interval_silence=indextts_interval_silence, indextts_max_text_tokens=indextts_max_text_tokens,
        indextts_temperature=indextts_temperature, indextts_top_p=indextts_top_p,
        indextts_top_k=indextts_top_k, indextts_repetition_penalty=indextts_repetition_penalty,
        indextts_max_mel_tokens=indextts_max_mel_tokens, indextts_timeout=indextts_timeout,
        ifly_ws_url=ifly_ws_url, ifly_app_id=ifly_app_id, ifly_key=ifly_key, ifly_secret=ifly_secret,
        ifly_vcn=args.ifly_vcn, ifly_speed=args.ifly_speed, ifly_volume=args.ifly_volume, ifly_pitch=args.ifly_pitch,
        ifly_encoding=args.ifly_encoding, ifly_sr=args.ifly_sr, ifly_channels=args.ifly_channels, ifly_bit_depth=args.ifly_bit_depth,
        ifly_oral_level=args.ifly_oral_level, ifly_stop_split=args.ifly_stop_split, ifly_remain=args.ifly_remain,
        ifly_by_sent=args.ifly_by_sent, ifly_pause_ms=args.ifly_pause_ms,
        subtitle=(args.subtitle=="on"),
        fade=args.fade, fps=args.fps,
        target_width=args.target_width, target_height=args.target_height,
        pad_silence=args.pad_silence, on_empty=args.on_empty,
        workers=args.workers,
        fast_concat=(args.fast_concat=="on"),
        enc_workers=args.enc_workers, enc_threads=args.enc_threads,
        enc_preset=args.enc_preset, enc_crf=args.enc_crf,
        debug=args.debug, force_ocr=args.force_ocr,
        skip_stems=skip_stems_value, skip_silence_sec=skip_silence_value,
        filter_keywords=filter_keywords_tuple,
    )

if __name__ == "__main__":
    main()
