from flask import Flask, request, jsonify, send_file # Added send_file
import tempfile, os
import sys

# Add the parent directory to the Python path so watermark_dwt and utils can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watermark_dwt import embed_watermark # Your custom module
from utils import Timer # Your custom module

app = Flask(__name__)

# This is the Vercel handler function
def handler(event, context):
    with app.request_context(event):
        response = app.full_dispatch_request()
        return response.get_data(), response.status_code, dict(response.headers)

@app.route("/embed", methods=["POST"])
def embed():
    cover = request.files.get("cover")
    watermark = request.files.get("watermark")
    alpha = float(request.form.get("alpha", 0.4))
    level = int(request.form.get("level", 1))

    if not cover or not watermark:
        return jsonify({"error": "Missing files"}), 400

    # Using tempfile.NamedTemporaryFile is good, but ensure it's in a writable directory.
    # Vercel functions provide /tmp for this.
    # The delete=False is good for ensuring the file exists for subsequent reads before explicit unlink.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir="/tmp") as f_cover, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir="/tmp") as f_wm, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir="/tmp") as f_out:

        cover.save(f_cover.name)
        watermark.save(f_wm.name)

        try:
            # Use Timer from utils module
            with Timer() as t:
                stego_img, psnr = embed_watermark(
                    f_cover.name,
                    f_wm.name,
                    f_out.name, # Save to the temporary output file
                    alpha=alpha,
                    level=level
                )
            # send_file automatically handles cleanup of the temporary file after sending
            return send_file(
                f_out.name,
                mimetype="image/png",
                as_attachment=True,
                download_name="stego.png",
                headers={"PSNR": str(psnr), "X-Processing-Time": str(t.elapsed)} # Add processing time for debug
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up temporary files. send_file might do this for f_out, but good to be explicit for others.
            os.unlink(f_cover.name)
            os.unlink(f_wm.name)
            # If send_file is used, f_out might still be in use by the OS, so explicit unlink might fail immediately.
            # Vercel's /tmp is cleaned between invocations anyway.
            if os.path.exists(f_out.name):
                os.unlink(f_out.name)

# Remove if __name__ == "__main__": app.run() as Vercel doesn't use it.
