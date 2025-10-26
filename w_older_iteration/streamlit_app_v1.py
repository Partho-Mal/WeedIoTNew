import streamlit as st
import numpy as np
import cv2
import io
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt

from src.models.segment_stub import segment
from src.drone_sim import simulate_drone_from_image
from src.compression import mask_to_heatmap
from src.http_client import send_heatmap_http
from src.aggregator import aggregate_list_of_heatmaps

# Streamlit setup
st.set_page_config(layout="wide", page_title="🌱 WeedIoT Dashboard")
st.title("🌾 WeedIoT — Upload → Segment → Simulate → Send → Aggregate")

# ======================
# 1️⃣ Left Panel — Input
# ======================
col_left, col_right = st.columns([2, 1])

with col_left:
    st.header("1) Upload image (any crop image)")
    uploaded = st.file_uploader("Upload crop image", type=["jpg", "jpeg", "png"])
    method = st.selectbox("Segmentation method", ["ndvi", "color"], index=0)
    threshold = st.slider("Segmentation threshold", 0.0, 1.0, 0.12, 0.01)

    st.markdown("**Drone simulation settings**")
    rows = st.slider("Drone rows", 1, 4, 2)
    cols = st.slider("Drone cols", 1, 6, 3)
    num_drones = rows * cols
    drop_prob = st.slider("Packet loss probability", 0.0, 0.5, 0.1, step=0.05)

    send_mode = st.radio("Transmission mode", ["HTTP (default)", "MQTT (optional)"])
    run_btn = st.button("🚀 Run simulation")

# ==========================
# 2️⃣ Right Panel — Results
# ==========================
with col_right:
    st.header("2) Farm-wide aggregated map")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    agg_path = results_dir / "aggregated.json"

    def load_aggregated_data(path: Path):
        if not path.exists() or path.stat().st_size == 0:
            path.write_text(json.dumps({"agg": []}, indent=2))
            return []
        try:
            data = json.loads(path.read_text())
            return data.get("agg", [])
        except json.JSONDecodeError:
            path.write_text(json.dumps({"agg": []}, indent=2))
            return []

    agg = load_aggregated_data(agg_path)

    if len(agg) > 0:
        agg_np = np.array(agg)
        st.image(agg_np, caption="Aggregated weed density (0..1)", clamp=True, use_column_width=True)
    else:
        st.info("No aggregated map yet. Run simulation to see results.")

# ==========================
# 3️⃣ Main Simulation Logic
# ==========================
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Failed to read image file.")
    else:
        # Resize large images
        h, w = img.shape[:2]
        max_side = 512
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        st.subheader("Preview")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_column_width=True)

        mask, aux = segment(img, method=method, threshold=threshold)
        st.subheader("Segmentation Mask (Weed = white)")
        st.image(mask, caption="Predicted weed mask", clamp=True)

        if run_btn:
            st.info("🛰️ Simulating drones and sending heatmaps...")
            heatmaps = simulate_drone_from_image(img, num_drones=num_drones,
                                                 drop_prob=drop_prob,
                                                 seg_method=method,
                                                 threshold=threshold)

            # ========== Simple Drone Animation ==========
            st.subheader("Drone Transmission Animation")
            animation_placeholder = st.empty()
            drone_positions = [(c, r) for r in range(rows) for c in range(cols)]
            base_pos = (cols + 1, rows / 2)

            for step in np.linspace(0, 1, 12):
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.set_xlim(-1, cols + 2)
                ax.set_ylim(-1, rows + 1)
                ax.axis('off')

                for i, (x, y) in enumerate(drone_positions):
                    ax.scatter(x, y, c='blue', s=200, marker='^', edgecolors='black')
                    if heatmaps[i] is not None:
                        x2 = x + step * (base_pos[0] - x)
                        y2 = y + step * (base_pos[1] - y)
                        ax.plot([x, x2], [y, y2], c='green', alpha=0.6, linewidth=2)
                        ax.scatter(x2, y2, c='lime', s=50)
                    else:
                        ax.text(x, y - 0.3, "X", color='red', fontsize=12, ha='center')
                ax.scatter(*base_pos, c='red', s=300, marker='s')
                ax.text(base_pos[0], base_pos[1] + 0.3, "Server", color='black', ha='center')
                animation_placeholder.pyplot(fig)
                plt.close(fig)
                time.sleep(0.05)

            # ============================================

            received = 0
            for idx, heat in enumerate(heatmaps):
                if heat is None:
                    st.write(f"Drone #{idx+1}: ❌ Packet dropped")
                else:
                    received += 1
                    ok = send_heatmap_http(heat, drone_id=f"drone_{idx+1}") if send_mode == "HTTP (default)" else True
                    st.write(f"Drone #{idx+1}: ✅ Sent → {ok}")

            st.success(f"Simulation complete: {received}/{num_drones} packets delivered.")

            # Wait for server aggregation
            time.sleep(1.0)
            agg = load_aggregated_data(agg_path)

            if len(agg) > 0:
                st.subheader("🌍 Updated Farm-wide Heatmap")
                st.image(np.array(agg), caption="Aggregated weed density", clamp=True, use_column_width=True)
            else:
                st.warning("No aggregation data yet — check if Flask server is running and receiving uploads.")

            # Transmission metrics
            st.markdown("---")
            st.header("Transmission Metrics")
            orig_size = uploaded.size * num_drones
            compressed_nbytes = sum([h.nbytes for h in heatmaps if h is not None])
            saved = (1.0 - (compressed_nbytes / (orig_size + 1e-9))) * 100.0
            st.metric("Packets sent", num_drones)
            st.metric("Packets received", received)
            st.metric("Original total size", f"{orig_size/1024:.2f} KB")
            st.metric("Compressed total", f"{compressed_nbytes/1024:.2f} KB")
            st.metric("Bandwidth saved", f"{saved:.1f}%")
