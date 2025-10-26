# streamlit_app.py
"""
Run with:
    source venv/bin/activate
    streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import cv2
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.segment_stub import segment
from src.drone_sim import simulate_drone_from_image
from src.http_client import send_heatmap_http
from src.aggregator import aggregate_list_of_heatmaps
from src.compression import mask_to_heatmap

# ======================================
# Streamlit App Configuration
# ======================================
st.set_page_config(layout="wide", page_title="🌾 WeedIoT Dashboard")

st.markdown("""
<style>
    .block-container {
        max-width: 1200px;
        margin: auto;
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌾 WeedIoT — Upload → Segment → Simulate → Send → Aggregate")

# ======================================
# Session State Setup
# ======================================
for key in ["last_method", "last_threshold", "last_image", "heatmaps"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "heatmaps" else []

# ======================================
# Helper Function
# ======================================
def load_aggregated_data(path: Path):
    """Safely load aggregated farm data."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        data = json.loads(path.read_text())
        return data.get("agg", [])
    except Exception:
        return []

# ======================================
# Layout Setup
# ======================================
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
agg_path = results_dir / "aggregated.json"

col_left, col_right = st.columns([2, 1])

# ======================================
# LEFT PANEL – User Input
# ======================================
with col_left:
    st.header("1️⃣ Upload and Configure")
    uploaded = st.file_uploader("Upload a crop image", type=["jpg", "jpeg", "png"])

    method = st.selectbox(
        "Weed Detection Method",
        ["color", "size_filter", "texture", "ndvi"],
        help="Select the segmentation approach for weed detection."
    )

    threshold = st.slider(
        "Segmentation threshold",
        0.0, 1.0, 0.12, 0.01,
        help="Lower = more sensitive (detects smaller weeds)."
    )

    st.markdown("**Drone Simulation Settings**")
    rows = st.slider("Drone rows", 1, 4, 2)
    cols = st.slider("Drone cols", 1, 6, 3)
    num_drones = rows * cols
    drop_prob = st.slider("Packet loss probability", 0.0, 0.5, 0.1, step=0.05)

    send_mode = st.radio("Transmission mode", ["HTTP (default)", "MQTT (optional)"])
    run_btn = st.button("🚀 Run Simulation")

# ======================================
# RIGHT PANEL – Aggregation Display
# ======================================
# with col_right:
#     st.header("2️⃣ Farm-wide Aggregated Map")
#     agg_data = load_aggregated_data(agg_path)
#     if len(agg_data) > 0:
#         agg_np = np.array(agg_data)
#         fig_agg, ax_agg = plt.subplots(figsize=(4, 4))
#         cax = ax_agg.matshow(agg_np, cmap="Greens")
#         ax_agg.set_title("Aggregated Weed Density", pad=10)
#         ax_agg.axis('off')
#         fig_agg.colorbar(cax, label="Density (0–1)", fraction=0.046)
#         st.pyplot(fig_agg)
#         plt.close(fig_agg)
#     else:
#         st.info("No aggregated map yet — run simulation to see results.")
# ======================================
# RIGHT PANEL – Aggregation Display
# ======================================
with col_right:
    if uploaded:
        # Convert uploaded image
        uploaded.seek(0)
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("❌ Failed to read image file.")
            st.stop()

        # Resize for consistent display
        h, w = img.shape[:2]
        max_side = 512
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # Display uploaded image
        st.subheader("📷 Uploaded Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input Image", width=350)

        # Run segmentation
        mask, aux = segment(img, method=method, threshold=threshold)
        st.session_state.mask = mask

        # ✅ Segmentation result in right column
        st.subheader("🧠 Segmentation Result (Weed = White)")
        fig_mask, ax_mask = plt.subplots(figsize=(5, 4))
        ax_mask.imshow(mask, cmap="gray")
        ax_mask.set_title("Predicted Weed Mask")
        ax_mask.axis('off')
        st.pyplot(fig_mask)
        plt.close(fig_mask)

          # Reset cache if parameters changed
        if (
            st.session_state.last_method != method or
            st.session_state.last_threshold != threshold or
            st.session_state.last_image != uploaded.name
        ):
            st.session_state.heatmaps = []
            st.session_state.last_method = method
            st.session_state.last_threshold = threshold
            st.session_state.last_image = uploaded.name

# ======================================
# IMAGE PROCESSING
# ======================================


  

 # ======================================
# SIMULATION (runs in LEFT/CENTER PANEL)
# ======================================
with col_left:
    if run_btn:
        st.info("🛰️ Simulating all drones...")

        # Run all at once (no per-drone loop)
        heatmaps = simulate_drone_from_image(
            img,
            num_drones=num_drones,
            drop_prob=drop_prob,
            seg_method=method,
            threshold=threshold,
        )

        st.session_state.heatmaps = heatmaps
        st.success(f"Drone simulation completed with {num_drones} drones.")
    else:
        heatmaps = st.session_state.heatmaps or []

        # ======================================
    # VISUALIZATION – Drone Overview (Animated)
    # ======================================
    if heatmaps:
        st.subheader("🚁 Drone Transmission Visualization")

        animation_placeholder = st.empty()

        # Arrange drone positions in grid
        drone_positions = [(c * 1.5 + 1, r * 1.5 + 1) for r in range(rows) for c in range(cols)]
        base_pos = (cols * 1.5 + 2, (rows * 1.5) / 2 + 0.5)

        # Animate packet movement
        for step in np.linspace(0, 1, 10):  # 10 animation frames
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(-0.5, cols * 1.5 + 3)
            ax.set_ylim(-0.5, rows * 1.5 + 2)
            ax.axis("off")

            # Draw drones
            for i, (x, y) in enumerate(drone_positions):
                ax.scatter(x, y, s=200, c='blue', marker='^', edgecolors='black', linewidths=1.5)
                ax.text(x, y - 0.3, f"D{i+1}", ha='center', fontsize=8, weight='bold')

            # Draw base station
            ax.scatter(*base_pos, s=350, c='red', marker='s', edgecolors='black', linewidths=1.5)
            ax.text(base_pos[0], base_pos[1] + 0.4, "Base Station", ha='center', fontsize=9, weight='bold')

            # Draw packets moving
            for i, (x, y) in enumerate(drone_positions):
                if heatmaps[i] is not None:
                    # Moving packet (green dot)
                    px = x + step * (base_pos[0] - x)
                    py = y + step * (base_pos[1] - y)
                    ax.plot([x, px], [y, py], c='green', linewidth=1.8, alpha=0.6)
                    ax.scatter(px, py, s=60, c='limegreen', edgecolors='darkgreen', linewidths=1)
                else:
                    # Dropped packet
                    ax.scatter(x, y, s=120, c='red', marker='x', linewidths=2)
                    ax.text(x, y - 0.4, "Dropped", ha='center', fontsize=7, color='red', style='italic')

            animation_placeholder.pyplot(fig)
            plt.close(fig)
            time.sleep(0.12)

        # # After animation, show final static frame
        # fig_final, ax_final = plt.subplots(figsize=(8, 6))
        # ax_final.set_xlim(-0.5, cols * 1.5 + 3)
        # ax_final.set_ylim(-0.5, rows * 1.5 + 2)
        # ax_final.axis("off")

        # for i, (x, y) in enumerate(drone_positions):
        #     ax_final.scatter(x, y, s=200, c='blue', marker='^', edgecolors='black', linewidths=1.5)
        #     ax_final.text(x, y - 0.3, f"D{i+1}", ha='center', fontsize=8, weight='bold')

        #     if heatmaps[i] is not None:
        #         ax_final.plot([x, base_pos[0]], [y, base_pos[1]], c='green', linewidth=1.8, alpha=0.6)
        #     else:
        #         ax_final.scatter(x, y, s=120, c='red', marker='x', linewidths=2)
        #         ax_final.text(x, y - 0.4, "Dropped", ha='center', fontsize=7, color='red', style='italic')

        # ax_final.scatter(*base_pos, s=350, c='red', marker='s', edgecolors='black', linewidths=1.5)
        # ax_final.text(base_pos[0], base_pos[1] + 0.4, "Base Station", ha='center', fontsize=9, weight='bold')

        # st.pyplot(fig_final)
        # plt.close(fig_final)

    # ======================================
    # TRANSMISSION & METRICS
    # ======================================
    if heatmaps:
        st.subheader("📡 Transmission Status")
        received = 0

        for idx, heat in enumerate(heatmaps):
            if heat is None:
                st.write(f"Drone #{idx+1}: ❌ Packet dropped")
                continue

            ok = False
            if send_mode == "HTTP (default)":
                ok = send_heatmap_http(heat, drone_id=f"drone_{idx+1}")
            else:
                try:
                    from src.mqtt_client import send_heatmap_mqtt
                    ok = send_heatmap_mqtt(heat, drone_id=f"drone_{idx+1}")
                except Exception:
                    ok = False

            if ok:
                received += 1
            st.write(f"Drone #{idx+1}: {'✅ Sent' if ok else '⚠️ Failed'}")

        st.success(f"Simulation complete: {received}/{num_drones} packets delivered.")

        # Aggregated map reload
        time.sleep(0.8)
        agg_data = load_aggregated_data(agg_path)
        if len(agg_data) > 0:
            st.subheader("🌍 Updated Farm-wide Heatmap")
            agg_np = np.array(agg_data)
            fig_updated, ax_updated = plt.subplots(figsize=(5, 4))
            cax = ax_updated.matshow(agg_np, cmap="Greens")
            ax_updated.set_title("Updated Aggregated Weed Density", pad=10)
            ax_updated.axis('off')
            fig_updated.colorbar(cax, label="Weed Density (0–1)")
            st.pyplot(fig_updated)
            plt.close(fig_updated)
        else:
            st.warning("No aggregation data found. Ensure Flask server is running.")

        # ======================================
        # TRANSMISSION METRICS
        # ======================================
        st.markdown("---")
        st.header("📊 Transmission Metrics")

        original_size = img.nbytes * num_drones
        compressed_nbytes = sum([h.nbytes for h in heatmaps if h is not None])
        saved = (1.0 - (compressed_nbytes / (original_size + 1e-9))) * 100.0

        col1, col2, col3 = st.columns(3)
        col1.metric("Packets Sent", num_drones)
        col1.metric("Packets Received", received)
        col2.metric("Original Size", f"{original_size/1024:.2f} KB")
        col2.metric("Compressed Size", f"{compressed_nbytes/1024:.2f} KB")
        col3.metric("Bandwidth Saved", f"{saved:.1f}%")
        col3.metric("Success Rate", f"{(received / num_drones) * 100:.1f}%")
