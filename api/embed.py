from flask import Flask, request, jsonify
from watermark_dwt import embed_watermark
import tempfile, os

app = Flask(__name__)

@app.route("/embed", methods=["POST"])
def embed():
    cover = request.files.get("cover")
    watermark = request.files.get("watermark")
    alpha = float(request.form.get("alpha", 0.4))
    level = int(request.form.get("level", 1))

    if not cover or not watermark:
        return jsonify({"error": "Missing files"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f_cover, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f_wm, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f_out:

        cover.save(f_cover.name)
        watermark.save(f_wm.name)

        try:
            stego_img, psnr = embed_watermark(
                f_cover.name,
                f_wm.name,
                f_out.name,
                alpha=alpha,
                level=level
            )
            with open(f_out.name, "rb") as f:
                stego_bytes = f.read()
        finally:
            os.unlink(f_cover.name)
            os.unlink(f_wm.name)
            os.unlink(f_out.name)

    return (stego_bytes, 200, {
        "Content-Type": "image/png",
        "Content-Disposition": "attachment; filename=stego.png",
        "PSNR": str(psnr)
    })

if __name__ == "__main__":
    app.run()
