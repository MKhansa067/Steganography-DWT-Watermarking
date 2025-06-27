import cv2
import numpy as np
import pywt
from PIL import Image, ImageDraw, ImageFont
import math
import os

def _calc_psnr(original: np.ndarray, stego: np.ndarray) -> float:
    mse = np.mean((original.astype(np.float32) - stego.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def _to_grayscale(img_path: str) -> np.ndarray:
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

def _text_to_image(text: str, size=(256, 256)) -> np.ndarray:
    """Buat citra biner dari teks (untuk watermark teks)."""
    img = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        w, h = draw.textsize(text, font=font)

    draw.text(((size[0] - w) // 2, (size[1] - h) // 2),
              text, fill=0, font=font)
    return np.array(img)

def embed_watermark(
    cover_path: str,
    wm_source: str,
    stego_path: str = "stego.png",
    alpha: float = 0.4,
    level: int = 1
):
    """
    Tambah watermark (gambar atau teks) ke dalam cover image
    menggunakan DWT level-`level` hanya pada sub-band HH.
    """

    cover = _to_grayscale(cover_path).astype(np.float32)

    if os.path.isfile(wm_source) and wm_source.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp")
    ):
        wm = _to_grayscale(wm_source)
    else:
        wm = _text_to_image(wm_source)

    coeffs = pywt.wavedec2(cover, 'haar', level=level)
    LL, detail_coeffs = coeffs[0], list(coeffs[1:])
    LH, HL, HH = detail_coeffs[0]
    target_shape = HH.shape

    wm_resized = cv2.resize(wm, (target_shape[1], target_shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    wm_bin = (wm_resized > 127).astype(np.float32)

    HH_emb = HH + alpha * wm_bin
    detail_coeffs[0] = (LH, HL, HH_emb)
    coeffs_new = [LL] + detail_coeffs

    stego = pywt.waverec2(coeffs_new, 'haar')
    stego_clipped = np.clip(stego, 0, 255).astype(np.uint8)
    cv2.imwrite(stego_path, stego_clipped)

    psnr_val = _calc_psnr(cover.astype(np.uint8), stego_clipped)
    return stego_clipped, psnr_val

def extract_watermark(stego_path: str, alpha: float = 0.10, level: int = 1):
    stego  = _to_grayscale(stego_path).astype(np.float32)
    coeffs = pywt.wavedec2(stego, "haar", level=level)
    HH     = coeffs[1][2]
    wm_est = HH / alpha
    
    wm_uint8  = cv2.normalize(wm_est, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    wm_eq = cv2.equalizeHist(wm_uint8)
    _, wm_bin = cv2.threshold(wm_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return wm_bin

def correlation(original_wm: np.ndarray, recovered_wm: np.ndarray) -> float:
    """Hitung korelasi normalisasi −1..1 antara dua watermark biner."""
    o = original_wm.flatten().astype(np.float32)
    r = recovered_wm.flatten().astype(np.float32)
    o = (o - o.mean()) / (o.std() + 1e-8)
    r = (r - r.mean()) / (r.std() + 1e-8)
    return float(np.mean(o * r))