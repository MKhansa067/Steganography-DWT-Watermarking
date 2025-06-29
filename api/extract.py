from flask import Flask, request, jsonify, send_file
from watermark_dwt import extract_watermark
import tempfile, os, cv2

app = Flask(__name__)

@app.route("/extract", methods=["POST"])
def extract():
    stego = request.files.get("stego")
    alpha = float(request.form.get("alpha", 0.4))
    level = int(request.form.get("level", 1))

    if not stego:
        return jsonify({"error": "Missing stego image"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f_stego, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f_out:

        stego.save(f_stego.name)

        try:
            wm_img = extract_watermark(f_stego.name, alpha=alpha, level=level)
            cv2.imwrite(f_out.name, wm_img)
        finally:
            os.unlink(f_stego.name)

    return send_file(f_out.name, mimetype="image/png", as_attachment=True, download_name="watermark.png")

if __name__ == "__main__":
    app.run()
