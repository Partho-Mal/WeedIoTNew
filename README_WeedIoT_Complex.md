# 🌱 WeedIoT — Drone-based Precision Farming Simulator

WeedIoT is a **precision agriculture simulation platform** that demonstrates **weed detection, drone-based heatmap generation, IoT transmission (HTTP/MQTT), and farm-wide aggregation**. The system combines image processing, segmentation algorithms, and network simulation to provide realistic scenarios for automated weed detection and management.

---

# 1. Project Overview

The workflow of WeedIoT:

1. Upload an image of a crop field (containing crops and weeds).
2. Segment weeds using multiple detection algorithms:

   * NDVI-based vegetation detection
   * Color and size filtering
   * Texture-based segmentation
   * Row-crop specific weed detection
3. Simulate multiple drones:

   * Each drone captures and processes the field image independently.
   * Generates **compressed heatmaps** representing weed density.
   * Simulates packet loss to mimic real-world network issues.
4. Transmit heatmaps via **HTTP** or **MQTT** to a central server.
5. Aggregate all heatmaps into a **farm-wide weed density map**.
6. Display **transmission metrics**, including compression efficiency and success rate.

---

# 2. Key Modules

## 2.1 Segmentation (src/models/segment_stub.py, src/preprocessing.py)

* **NDVI-based detection**: Computes a proxy vegetation index using the green and red channels. Detects all vegetation without distinguishing weeds from crops.
* **Color + size filtering**: Uses HSV thresholds and contour filtering to isolate small, irregular vegetation patches (weeds) while ignoring larger crops.
* **Texture-based detection**: Uses local variance analysis to detect irregular patterns typical of weeds.
* **Row-crop analysis**: Identifies crop rows and detects weeds growing between them.
* **Auxiliary functions**:

  * compute_ndvi_from_bgr()
  * threshold_mask_from_ndvi()
  * detect_weeds_by_size_color()
  * detect_weeds_texture_based()

## 2.2 Drone Simulation (src/drone_sim.py)

* Simulates multiple drones scanning the same image.
* Each drone:

  1. Adds slight perturbation (brightness, noise) to mimic real-world conditions.
  2. Segments weeds using the chosen method.
  3. Compresses mask into a `20x20` heatmap grid (mask_to_heatmap).
  4. Simulates **packet loss** (probabilistic drop).

Function: `simulate_drone_from_image(image_bgr, num_drones, drop_prob, seg_method, threshold)`

## 2.3 Heatmap Compression (src/compression.py)

* Converts binary masks (0 = background, 255 = weeds) into **density heatmaps**.
* Computes average weed density per grid cell.
* Saves **bandwidth** by sending compressed heatmaps instead of full images.

Function: `mask_to_heatmap(mask_np, grid_size=20)`

## 2.4 Transmission (src/http_client.py, src/mqtt_client.py)

* **HTTP Client**: Sends heatmap JSON to a Flask server.
* **MQTT Client**: Publishes heatmap to broker topic (`farm/weed_data`).

Example HTTP payload:

{
"drone_id": "drone_1",
"heatmap": [[0.0, 0.2, ...], [...]]
}

## 2.5 Aggregation (src/aggregator.py)

* Aggregates heatmaps from multiple drones.
* Computes **average weed density** per cell.
* Persists aggregated farm map in `results/aggregated.json`.

Function: `aggregate_list_of_heatmaps(list_of_heatmaps)`

## 2.6 Server (src/server.py)

* Flask server receives heatmaps via HTTP POST.
* Aggregates data and returns farm-wide weed density via `/get_aggregate`.

## 2.7 Dashboard (streamlit_app.py)

* Streamlit UI allows:

  * Image upload
  * Selection of segmentation method
  * Drone simulation configuration (rows, columns, packet loss)
  * Real-time **drone transmission animation**
  * Display of metrics:

    * Packets Sent / Received
    * Original Size / Compressed Size
    * Bandwidth Saved
    * Success Rate
  * Farm-wide aggregated heatmap visualization

---

# 3. Algorithms Used

1. **NDVI Computation**:

NDVI = (G - R) / (G + R + epsilon), where G = green channel, R = red channel

2. **Heatmap Aggregation**:

Heatmap_agg(i,j) = (1/N) * sum_{k=1}^N Heatmap_k(i,j)

3. **Weed Segmentation Approaches**:

   * Color filtering + size threshold
   * Texture irregularity detection using local variance
   * Row-crop analysis for between-row weed detection

4. **Drone Simulation & Packet Loss**:

   * Each drone has probability `p_drop` of dropping a packet
   * Successful heatmaps aggregated for farm-wide density map

---

# 4. Transmission Metrics

| Metric           | Description                                         |
| ---------------- | --------------------------------------------------- |
| Packets Sent     | Total number of simulated drones                    |
| Packets Received | Successfully delivered heatmaps                     |
| Original Size    | Size of raw images in memory                        |
| Compressed Size  | Size of heatmaps sent                               |
| Bandwidth Saved  | Compression efficiency: 1 - (compressed / original) |
| Success Rate     | Delivery success: (received / sent) * 100%          |

---

# 5. Installation & Usage

1. Clone repository:

git clone <repo-url>
cd WeedIoT

2. Install dependencies:

pip install -r requirements.txt

3. Start Flask server:

python src/server.py 
or 
python3 -m src.server

4. Run Streamlit dashboard:

streamlit run streamlit_app.py

5. Upload crop images and configure drone simulation.

---

# 6. Notes

* Supports **HTTP and MQTT** transmission.
* Adjustable drone grid and packet loss probability for network simulation.
* Aggregated heatmap enables **precision herbicide spraying planning**.
* Visual animations help understand **drone coordination and packet reliability**.

---

# 7. References

* OpenCV for image processing
* NumPy for numerical computation
* Streamlit for interactive dashboard
* Flask for server
* MQTT (Paho) for IoT simulation
* NDVI and weed segmentation literature

