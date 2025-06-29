from flask import Flask, request, jsonify, send_file
import tempfile, os
import sys

# Add the parent directory to the Python path so watermark_dwt can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watermark_dwt import extract_watermark # Your custom module
import cv2 # cv2 is needed here as well for imwrite

app = Flask(__name__)

# This is the Vercel handler function
def handler(event, context):
    with app.request_context(event):
        response = app.full_dispatch_request()
        return response.get_data(), response.status_code, dict(response.headers)

@app.route("/extract", methods=["POST"])
def extract():
    stego = request.files.get("stego")
    alpha = float(request.form.get("alpha", 0.4))
    level = int(request.form.get("level", 1))

    if not stego:
        return jsonify({"error": "Missing stego image"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir="/tmp") as f_stego, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir="/tmp") as f_out: # f_out for the extracted watermark

        stego.save(f_stego.name)

        try:
            wm_img = extract_watermark(f_stego.name, alpha=alpha, level=level)
            cv2.imwrite(f_out.name, wm_img) # Save extracted watermark to f_out

            return send_file(f_out.name, mimetype="image/png", as_attachment=True, download_name="watermark.png")
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            os.unlink(f_stego.name)
            if os.path.exists(f_out.name): # Ensure f_out is cleaned up after send_file, if it fails to do so
                os.unlink(f_out.name)

# Remove if __name__ == "__main__": app.run() as Vercel doesn't use it.
