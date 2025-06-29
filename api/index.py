from flask import Flask, request, send_file, render_template
import os, cv2, numpy as np, tempfile
from watermark_dwt import embed_watermark, extract_watermark

app = Flask(__name__, template_folder="../templates")

UPLOAD_DIR = tempfile.gettempdir()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/embed", methods=["POST"])
def embed():
    cover      = request.files["cover"]
    watermark  = request.files["watermark"]

    cover_path     = os.path.join(UPLOAD_DIR, "cover.png")
    watermark_path = os.path.join(UPLOAD_DIR, "wm.png")
    cover.save(cover_path);  watermark.save(watermark_path)

    stego_img, _ = embed_watermark(cover_path, watermark_path,
                                   os.path.join(UPLOAD_DIR, "stego.png"))
    return send_file(os.path.join(UPLOAD_DIR, "stego.png"),
                     mimetype="image/png", as_attachment=True,
                     download_name="stego.png")

@app.route("/extract", methods=["POST"])
def extract():
    stego = request.files["stego"]
    stego_path = os.path.join(UPLOAD_DIR, "stego_up.png")
    stego.save(stego_path)

    rec = extract_watermark(stego_path)
    rec_path = os.path.join(UPLOAD_DIR, "recovered.png")
    cv2.imwrite(rec_path, rec)
    return send_file(rec_path, mimetype="image/png",
                     as_attachment=True, download_name="watermark.png")
