# src/server.py

"""
    run with 
    python3 -m src.server 
"""
from flask import Flask, request, jsonify, render_template_string
import numpy as np
from src.preprocessing import combined_weed_heatmap
from src.aggregator import aggregate_list_of_heatmaps, persist_aggregate
import json
from pathlib import Path
import sys
import os
import io
import base64
from matplotlib import pyplot as plt  # <-- Added import

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)

# in-memory store of recent heatmaps (for demo)
RECENT = []

@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    if not data or "heatmap" not in data:
        return jsonify({"error": "no heatmap"}), 400
    heat = np.array(data["heatmap"], dtype=float)
    RECENT.append(heat)
    # keep small history
    if len(RECENT) > 50:
        RECENT.pop(0)
    farm = aggregate_list_of_heatmaps(RECENT)
    if farm is not None:
        persist_aggregate(farm)
    return jsonify({"status": "ok"}), 200

@app.route("/get_aggregate", methods=["GET"])
def get_aggregate():
    ag = Path("results/aggregated.json")
    if not ag.exists():
        return jsonify({"agg": None})
    return ag.read_text(), 200, {"Content-Type": "application/json"}

@app.route("/upload_image", methods=["POST"])
def upload_image():
    data = request.get_json()
    if not data or "image_base64" not in data:
        return jsonify({"error": "no image provided"}), 400

    import cv2
    import base64
    import numpy as np
    from io import BytesIO

    # Decode base64
    img_bytes = base64.b64decode(data["image_base64"])
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Compute combined weed heatmap
    heat = combined_weed_heatmap(img_bgr)
    RECENT.append(heat)
    
    # Keep small history
    if len(RECENT) > 50:
        RECENT.pop(0)

    # Persist aggregate
    farm = aggregate_list_of_heatmaps(RECENT)
    if farm is not None:
        persist_aggregate(farm)

    return jsonify({"status": "ok"}), 200


@app.route("/")
def home():
    """
    Simple visualization page for browser
    """
    total_packets = len(RECENT)
    farm_agg = aggregate_list_of_heatmaps(RECENT)
    agg_img_html = ""
    
    if farm_agg is not None:
        # Convert heatmap to PNG base64 for browser display
        fig, ax = plt.subplots(figsize=(4, 4))
        cax = ax.matshow(farm_agg, cmap="Greens")
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        agg_img_html = f'<h3>Aggregated Weed Density</h3><img src="data:image/png;base64,{img_base64}" width="300">'
    
    html = f"""
    <html>
        <head><title>WeedIoT Server Dashboard</title></head>
        <body>
            <h1>WeedIoT Server Dashboard</h1>
            <p><b>Total Packets Received:</b> {total_packets}</p>
            {agg_img_html}
            <p>Use POST /upload to send heatmaps and GET /get_aggregate for JSON data.</p>
        </body>
    </html>
    """
    return render_template_string(html)

if __name__ == "__main__":
    # Run server: python src/server.py
    app.run(host="0.0.0.0", port=5000, debug=True)
