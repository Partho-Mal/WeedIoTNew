# streamlit_app.py
import streamlit as st
import numpy as np
import cv2
import io
from pathlib import Path
import json
import time
from src.models.segment_stub import segment
from src.drone_sim import simulate_drone_from_image
from src.aggregator import persist_aggregate, aggregate_list_of_heatmaps
from src.compression import mask_to_heatmap
from src.http_client import send_heatmap_http
# optional mqtt client: from src.mqtt_client import send_heatmap_mqtt


st.set_page_config(layout="wide", page_title="WeedIoT New Dashboard")
st.title("🌱 WeedIoT — Upload → Segment → Compress → Send → Aggregate")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.header("1) Upload image (any crop image)")
    uploaded = st.file_uploader("Upload a crop image (jpg/png)", type=["jpg", "jpeg", "png"])
    method = st.selectbox("Segmentation method", options=["ndvi", "color"], index=0)
    threshold = st.slider("Segmentation threshold", min_value=0.0, max_value=1.0, value=0.12, step=0.01)

    st.markdown("**Drone simulation** settings")
    rows = st.slider("Drone rows", 1, 4, 2)
    cols = st.slider("Drone cols", 1, 6, 3)
    num_drones = rows * cols
    drop_prob = st.slider("Packet loss probability", 0.0, 0.5, 0.1, step=0.05)

    st.markdown("---")
    send_mode = st.radio("Transmission mode", options=["HTTP (default)", "MQTT (optional)"])

    run_btn = st.button("Run simulation → segment → compress → send")
    st.markdown(" ")

with col_right:
    st.header("2) Farm-wide aggregated map")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    agg_path = results_dir / "aggregated.json"

    def load_aggregated_data(path: Path):
        """Safely load aggregated JSON or create an empty one if missing/corrupt."""
        if not path.exists() or path.stat().st_size == 0:
            path.write_text(json.dumps({"agg": []}, indent=2))
            return []
        try:
            data = json.loads(path.read_text())
            return data.get("agg", [])
        except json.JSONDecodeError:
            # File is corrupt → reset
            path.write_text(json.dumps({"agg": []}, indent=2))
            return []

    agg = load_aggregated_data(agg_path)

    if len(agg) > 0:
        agg_np = np.array(agg)
        st.image(agg_np, caption="Aggregated heatmap (0..1)", clamp=True, use_column_width=True)
    else:
        st.info("No aggregated map yet. Send heatmaps from the simulation.")

if uploaded:
    # read as OpenCV BGR
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Failed to read image")
    else:
        # Resize to reasonable size for speed
        h, w = img.shape[:2]
        max_side = 512
        if max(h,w) > max_side:
            scale = max_side / max(h,w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))

        st.subheader("Preview")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_column_width=True)

        # Run segmentation preview
        mask, aux = segment(img, method=method, threshold=threshold)
        st.subheader("Segmentation Mask (preview)")
        st.image(mask, caption="Predicted weed mask (255 = weed)", clamp=True)

        if run_btn:
            st.info("Simulating drones and sending compressed maps...")
            # simulate multi-drones
            heatmaps = simulate_drone_from_image(img, num_drones=num_drones, drop_prob=drop_prob, seg_method=method, threshold=threshold)
            # visualize per-drone status
            received = 0
            for idx, heat in enumerate(heatmaps):
                if heat is None:
                    st.write(f"Drone #{idx+1}: packet dropped")
                else:
                    received += 1
                    # send to server
                    if send_mode == "HTTP (default)":
                        ok = send_heatmap_http(heat, drone_id=f"drone_{idx+1}")
                    else:
                        # optional: requires running MQTT broker and MQTT sender implemented
                        try:
                            from src.mqtt_client import send_heatmap_mqtt
                            ok = send_heatmap_mqtt(heat, drone_id=f"drone_{idx+1}")
                        except Exception as e:
                            ok = False
                    st.write(f"Drone #{idx+1}: sent -> {ok}")

            st.success(f"Simulation done. {received}/{num_drones} packets received by server (simulated).")
            # fetch aggregated file and show
            time.sleep(0.5)
            if agg_path.exists():
                agg = json.loads(agg_path.read_text())["agg"]
                st.subheader("New Aggregated Farm Map")
                st.image(np.array(agg), caption="Aggregated heatmap (0..1)", clamp=True, use_column_width=True)

            # transmission metrics
            st.markdown("---")
            st.header("Transmission Metrics (simulated)")
            orig_size = uploaded.size * num_drones
            compressed_nbytes = sum([h.nbytes for h in heatmaps if h is not None])
            st.metric("Packets sent", num_drones)
            st.metric("Packets received", sum([1 for h in heatmaps if h is not None]))
            st.metric("Original size (sum of original images)", f"{orig_size/1024:.2f} KB")
            st.metric("Compressed size (sum of heatmaps)", f"{compressed_nbytes/1024:.2f} KB")
            saved = (1.0 - (compressed_nbytes / (orig_size + 1e-9))) * 100.0
            st.metric("Bandwidth saved (approx)", f"{saved:.2f}%")
