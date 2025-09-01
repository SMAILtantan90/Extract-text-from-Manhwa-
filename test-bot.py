#!/usr/bin/env python3

-- coding: utf-8 --

""" krbot_railway_ready.py نسخة معدلة من سكربت OCR لبوت ديسكورد، مهيّئة للعمل على خدمات مثل Railway. ميزات رئيسية:

يقرأ DISCORD_TOKEN من متغير بيئة (لا تضع التوكن داخل الكود)

استخدام pytesseract وأيضًا دعم اختياري لـ easyocr إذا كان مثبتًا

معالجة صور المانهوا الطويلة بتقسيمها إلى شرائح لتقليل استهلاك الذاكرة

اكتشاف مناطق النص باستخدام OpenCV (threshold -> dilation -> contours)

دمج المربعات وفلترة النتائج بدقة عالية

حفظ النتائج كملف نصي وملف JSON وإرسالها في الرسائل

لوغ/مراقبة أخطاء ووقت التنفيذ


ملاحظات نشر على Railway:

تأكد من أن تيسراكت مثبت في الحاوية (مثال: apt-get install -y tesseract-ocr tesseract-ocr-kor)

ضع التوكن في Environment Variable اسمه DISCORD_TOKEN

(اختياري) إذا أردت easyocr، أضف USE_EASYOCR=1 وركّب easyocr في requirements


"""

import os import io import json import math import time import tempfile import asyncio import logging from typing import List, Tuple, Optional

import aiohttp import discord from discord.ext import commands

from PIL import Image, ImageOps, ImageFilter import numpy as np

optional heavy libs

try: import cv2 except Exception: cv2 = None

try: import pytesseract except Exception: pytesseract = None

easyocr is optional fallback (pure-python-ish but heavy)

try: import easyocr except Exception: easyocr = None

------------------------- Configuration -------------------------

DEFAULT_LANG = os.getenv("OCR_LANG", "kor+eng")  # pytesseract style; change as needed DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")  # REQUIRED OUTPUT_FILE = os.getenv("OUTPUT_FILE", "extracted_text.txt") JSON_OUTPUT = os.getenv("JSON_OUTPUT", "extracted_text.json") TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract") USE_EASYOCR = os.getenv("USE_EASYOCR", "0") in ("1", "true", "True")

performance / heuristics

MAX_DIM = int(os.getenv("MAX_DIM", "3000"))  # cap largest image side to avoid OOM STRIP_HEIGHT = int(os.getenv("STRIP_HEIGHT", "2500"))  # split very long images STRIP_OVERLAP = int(os.getenv("STRIP_OVERLAP", "150")) MIN_AREA = int(os.getenv("MIN_AREA", "300"))  # ignore tiny contours CONF_THRESH = int(os.getenv("CONF_THRESH", "50")) GAP_FACTOR = float(os.getenv("GAP_FACTOR", "0.35"))

set pytesseract cmd if available

if pytesseract: pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s") logger = logging.getLogger("krbot")

------------------------- Utility helpers -------------------------

def pil_to_cv2(pil: Image.Image) -> np.ndarray: arr = np.array(pil.convert("RGB")) return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if cv2 else np.array(pil)

def cv2_to_pil(img: np.ndarray) -> Image.Image: if img is None: return None if img.ndim == 2: return Image.fromarray(img) b,g,r = cv2.split(img) return Image.fromarray(cv2.merge((r,g,b)))

def scale_image_keep_aspect(pil: Image.Image, max_side: int) -> Image.Image: w,h = pil.size if max(w,h) <= max_side: return pil scale = max_side / float(max(w,h)) nw, nh = int(wscale), int(hscale) return pil.resize((nw, nh), Image.LANCZOS)

------------------------- Text region detection (OpenCV) -------------------------

def detect_text_regions_cv(pil_img: Image.Image) -> List[Tuple[int,int,int,int]]: """اكتشاف مناطق النص بإستعمال OpenCV. يعيد قائمة من المربعات (x,y,w,h). خوارزمية مبسطة ولكن فعالة لمعظم صفحات المانهوا: grayscale -> adaptive thresh -> dilate -> contours. """ if cv2 is None: logger.warning("OpenCV غير مثبت — سيتم استعمال pytesseract الكامل كبديل") return []

img = pil_to_cv2(pil_img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# equalize or adaptive
# use morphological ops to join letters into boxes
blur = cv2.GaussianBlur(gray, (3,3), 0)
thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV, 25, 9)

# small closing to connect text components
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

# dilate a bit more to merge words into bubbles
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15,5))
dil = cv2.dilate(closed, kernel2, iterations=1)

contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = []
h_img, w_img = gray.shape
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    area = w*h
    if area < MIN_AREA:
        continue
    # filter extreme wide/flat areas (may be art), allow moderate aspect
    if w < 20 or h < 10:
        continue
    # clamp
    x = max(0,x); y = max(0,y)
    w = min(w_img - x, w); h = min(h_img - y, h)
    rects.append((x,y,w,h))

# merge overlapping/closenear rectangles greedily
merged = merge_boxes_greedy_rects(rects, gap_factor=GAP_FACTOR)
return merged

def rect_iou_simple(a, b): ax,ay,aw,ah = a; bx,by,bw,bh = b x1 = max(ax,bx); y1 = max(ay,by) x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh) inter = max(0, x2-x1) * max(0, y2-y1) union = awah + bwbh - inter return inter/union if union>0 else 0

def overlap_or_close_rect(a,b,gap_factor=0.35): if rect_iou_simple(a,b) > 0: return True ax,ay,aw,ah = a; bx,by,bw,bh = b right_a = ax+aw; right_b = bx+bw hor_gap = max(0, max(bx - right_a, ax - right_b)) bot_a = ay+ah; bot_b = by+bh ver_gap = max(0, max(by - bot_a, ay - bot_b)) max_dim = max(aw, ah, bw, bh, 1) return max(hor_gap, ver_gap) < max_dim * gap_factor

def merge_boxes_greedy_rects(rects: List[Tuple[int,int,int,int]], gap_factor: float = 0.35) -> List[Tuple[int,int,int,int]]: rects = rects.copy() used = [False]*len(rects) out = [] for i,r in enumerate(rects): if used[i]: continue mx,my,mw,mh = r used[i]=True changed=True while changed: changed=False for j,r2 in enumerate(rects): if used[j]: continue if overlap_or_close_rect((mx,my,mw,mh), r2, gap_factor): x2,y2,w2,h2 = r2 nx = min(mx, x2); ny = min(my, y2) nx2 = max(mx+mw, x2+w2); ny2 = max(my+mh, y2+h2) mx,my,mw,mh = nx,ny,nx2-nx,ny2-ny used[j]=True changed=True out.append((mx,my,mw,mh)) return out

------------------------- OCR helpers -------------------------

def ocr_with_pytesseract(pil_cropped: Image.Image, lang: str = DEFAULT_LANG) -> Tuple[str,float]: if pytesseract is None: return ("", 0.0) cfg = "--oem 3 --psm 6" try: data = pytesseract.image_to_data(pil_cropped, lang=lang, output_type=pytesseract.Output.DICT, config=cfg) except Exception: data = pytesseract.image_to_data(pil_cropped, lang=lang, output_type=pytesseract.Output.DICT) words = [ (data['text'][i] or "").strip() for i in range(len(data.get('text',[]))) ] confs = [] for c in data.get('conf',[]): try: confs.append(int(c)) except: pass text = " ".join([w for w in words if w]) avg_conf = (sum(confs)/len(confs)) if confs else 0.0 return (text, avg_conf)

def ocr_with_easyocr(pil_cropped: Image.Image, langs: Optional[List[str]]=None) -> Tuple[str,float]: if easyocr is None: return ("", 0.0) # easyocr expects numpy image BGR img = np.array(pil_cropped.convert('RGB'))[:, :, ::-1] reader = easyocr.Reader(langs or ['ko','en'], gpu=False) res = reader.readtext(img) texts = [r[1] for r in res] confs = [r[2] for r in res] text = " ".join(texts) avg = (sum(confs)/len(confs)) if confs else 0.0 return (text, avg)

def ocr_crop_best(pil_crop: Image.Image, lang: str = DEFAULT_LANG) -> Tuple[str, float]: # try multiple scales and both OCR engines (pytesseract first), return best by score scales = [1.0, 1.6, 2.4] best_text = "" best_score = -1.0 for s in scales: if s != 1.0: im = pil_crop.resize((int(pil_crop.widths), int(pil_crop.heights)), Image.LANCZOS) else: im = pil_crop gray = ImageOps.grayscale(im) gray = ImageOps.autocontrast(gray, cutoff=1) # pytesseract t, c = ocr_with_pytesseract(gray, lang=lang) if pytesseract else ("", 0.0) score = c * (len(t) + 1) if score > best_score and t.strip(): best_score = score; best_text = t # easyocr fallback if USE_EASYOCR and easyocr: t2, c2 = ocr_with_easyocr(gray, langs=None) score2 = c2 * (len(t2) + 1) if score2 > best_score and t2.strip(): best_score = score2; best_text = t2 return (best_text.strip(), best_score)

basic cleaner: keep hangul and ascii punctuation/numbers

import re _allowed_re = re.compile(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF0-9A-Za-z-.,!?'"():]+", re.UNICODE)

def clean_text_keep_korean(s: str) -> str: parts = _allowed_re.findall(s) t = " ".join(parts) t = re.sub(r"(.)\1{3,}", r"\1\1\1", t) t = t.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?") t = " ".join(t.split()) return t.strip()

------------------------- High-level extraction -------------------------

def extract_text_from_long_image(pil_img: Image.Image, lang: str = DEFAULT_LANG) -> List[dict]: """تتعامل مع صور المانهوا طويلة عبر تقسيمها إلى شرائح، استخراج مناطق النص ثم OCR لكل منطقة. تُعيد قائمة من dicts: {box:(x,y,w,h), text:str, score:float} """ start_t = time.time() # optionally scale down if extremely large to avoid OOM orig_w, orig_h = pil_img.size scale_factor = 1.0 if max(orig_w, orig_h) > MAX_DIM: scale_factor = MAX_DIM / float(max(orig_w, orig_h)) pil_img = pil_img.resize((int(orig_wscale_factor), int(orig_hscale_factor)), Image.LANCZOS) logger.info(f"scaled image by {scale_factor:.3f} -> {pil_img.size}")

# split into vertical strips if tall
w,h = pil_img.size
strips = []
if h > STRIP_HEIGHT:
    y = 0
    while y < h:
        y2 = min(h, y + STRIP_HEIGHT)
        # include overlap
        sy = max(0, y - STRIP_OVERLAP)
        ey = min(h, y2 + STRIP_OVERLAP)
        strips.append((sy, ey))
        y = y2
else:
    strips.append((0, h))

results = []
for si, (sy, ey) in enumerate(strips):
    crop = pil_img.crop((0, sy, w, ey))
    logger.info(f"processing strip {si+1}/{len(strips)} height={ey-sy}")
    # detect boxes
    rects = detect_text_regions_cv(crop)
    if not rects:
        # fallback: use pytesseract full-page word boxes
        if pytesseract:
            data = pytesseract.image_to_data(crop, lang=lang, output_type=pytesseract.Output.DICT)
            for i in range(len(data.get('text',[]))):
                txt = (data['text'][i] or "").strip()
                try:
                    conf = int(data['conf'][i])
                except:
                    conf = -1
                if not txt or conf < CONF_THRESH:
                    continue
                left = int(data['left'][i]); top = int(data['top'][i]); width = int(data['width'][i]); height = int(data['height'][i])
                rects.append((left, top, width, height))

    # normalize rects & OCR them
    for (x,y_r,w_r,h_r) in rects:
        # expand pad
        pad = max(6, int(min(w,h) * 0.03))
        x0 = max(0, x-pad); y0 = max(0, y_r-pad)
        x1 = min(w, x+w_r+pad); y1 = min(ey-sy, y_r+h_r+pad)
        crop_box = crop.crop((x0, y0, x1, y1))
        text, score = ocr_crop_best(crop_box, lang=lang)
        cleaned = clean_text_keep_korean(text)
        if not cleaned:
            continue
        # convert coords back to original image scale & offset
        if scale_factor != 1.0:
            inv_scale = 1.0/scale_factor
        else:
            inv_scale = 1.0
        orig_x = int((x0) * inv_scale)
        orig_y = int((y0 + sy) * inv_scale)
        orig_wbox = int((x1-x0) * inv_scale)
        orig_hbox = int((y1-y0) * inv_scale)
        results.append({
            'box': (orig_x, orig_y, orig_wbox, orig_hbox),
            'text': cleaned,
            'score': float(score)
        })

# sort top-to-bottom
results.sort(key=lambda r: (r['box'][1], r['box'][0]))
logger.info(f"extraction done: {len(results)} regions in {time.time()-start_t:.2f}s")
return results

------------------------- Discord bot -------------------------

intents = discord.Intents.default() intents.message_content = True bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event async def on_ready(): logger.info(f"Logged in as {bot.user} ({bot.user.id})") logger.info("Commands: !ocr (attach image) | !ocr url <link> | !ocr_show")

async def fetch_image_bytes_from_url(url: str) -> bytes: async with aiohttp.ClientSession() as session: async with session.get(url, timeout=60) as resp: if resp.status != 200: raise RuntimeError(f"HTTP {resp.status}") return await resp.read()

def chunk_text(s: str, max_len: int = 1900) -> List[str]: out = [] cur = [] cur_len = 0 for part in s.splitlines(): if cur_len + len(part) + 1 > max_len: out.append("\n".join(cur)) cur = [part] cur_len = len(part) else: cur.append(part) cur_len += len(part) + 1 if cur: out.append("\n".join(cur)) return out

@bot.command(name="ocr", help="استخراج النص من صورة مرفوعة أو من رابط. استعمل: !ocr أو !ocr url <link>") async def ocr_cmd(ctx, *, arg: str = None): await ctx.message.add_reaction("🔎") image_bytes = None lang = DEFAULT_LANG

# get image bytes
if arg and arg.strip().lower().startswith("url "):
    url = arg.strip()[4:].strip()
    try:
        image_bytes = await fetch_image_bytes_from_url(url)
    except Exception as e:
        await ctx.send(f"فشل تحميل الصورة: {e}")
        await ctx.message.remove_reaction("🔎", bot.user)
        return
elif ctx.message.attachments:
    attachment = ctx.message.attachments[0]
    try:
        image_bytes = await attachment.read()
    except Exception as e:
        await ctx.send(f"خطأ في قراءة المرفق: {e}")
        await ctx.message.remove_reaction("🔎", bot.user)
        return
else:
    await ctx.send("أرسل الأمر مع صورة مرفقة أو: `!ocr url <رابط الصورة>`")
    await ctx.message.remove_reaction("🔎", bot.user)
    return

# open image
try:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
except Exception as e:
    await ctx.send(f"خطأ في فتح الصورة: {e}")
    await ctx.message.remove_reaction("🔎", bot.user)
    return

# run extraction in thread pool
try:
    results = await asyncio.to_thread(extract_text_from_long_image, pil_img, lang)
except Exception as e:
    await ctx.send(f"خطأ أثناء الاستخراج: {e}")
    await ctx.message.remove_reaction("🔎", bot.user)
    return

if not results:
    await ctx.send("(لم يتم استخراج أي نص)")
    await ctx.message.remove_reaction("🔎", bot.user)
    return

# prepare outputs
lines = []
for r in results:
    x,y,w,h = r['box']
    lines.append(f"[{x},{y},{w},{h}]  {r['text']}")

# save to temp files
tdir = tempfile.mkdtemp(prefix="krbot_")
txt_path = os.path.join(tdir, OUTPUT_FILE)
json_path = os.path.join(tdir, JSON_OUTPUT)
try:
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
except Exception as e:
    logger.exception("failed saving outputs")

# send as chunks + attachments
joined = "\n".join(lines)
for chunk in chunk_text(joined):
    await ctx.send(f"```{chunk}```")
try:
    await ctx.send(file=discord.File(txt_path, filename=OUTPUT_FILE))
    await ctx.send(file=discord.File(json_path, filename=JSON_OUTPUT))
except Exception:
    pass

await ctx.message.remove_reaction("🔎", bot.user)

@bot.command(name="ocr_show", help="عرض محتوى ملف الناتج الأخير") async def ocr_show(ctx): if not os.path.exists(OUTPUT_FILE): await ctx.send("لا يوجد ملف محفوظ بعد.") return await ctx.send(file=discord.File(OUTPUT_FILE))

------------------------- Entrypoint -------------------------

if name == "main": if not DISCORD_TOKEN: logger.error("DISCORD_TOKEN غير موجود في متغيرات البيئة. ضع التوكن كـ DISCORD_TOKEN") raise SystemExit(1) # show which engines are available logger.info(f"OpenCV: {'yes' if cv2 else 'no'}, pytesseract: {'yes' if pytesseract else 'no'}, easyocr: {'yes' if easyocr else 'no'}") bot.run(DISCORD_TOKEN)

