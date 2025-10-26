# 🌱 WeedIoT — Precision Farming Simulator

WeedIoT is a precision farming simulation project that demonstrates **weed detection**, **multi-drone data transmission**, **compression**, and **farm-wide aggregation** using **image processing**, **NDVI/color segmentation**, and **IoT protocols** (HTTP & MQTT).

---

## 1️⃣ Project Overview

The project workflow:

1. Upload a crop image containing weeds.
2. Segment the weeds using **NDVI**, **color**, or other methods.
3. Simulate multiple drones scanning the field and generating heatmaps.
4. Compress the heatmaps to reduce data size for transmission.
5. Send the heatmaps to a central server (HTTP or MQTT).
6. Aggregate all drone data into a **farm-wide weed density map**.
7. Display **transmission metrics** and success statistics.

---

## 2️⃣ Segmentation

Segmentation identifies weed regions in the image:

- **NDVI-based:** Uses Red channel as NIR proxy to calculate a vegetation index.
- **Color-based:** Uses color thresholding to detect green weeds.
- **Texture/Height-based:** (Optional future implementation)

**Output:**  
A binary mask where **white = weed**, **black = background**.

---

## 3️⃣ Drone Simulation

- The dashboard allows selecting:
  - `Drone rows` and `Drone cols` → Number of drones = rows × cols
  - `Packet loss probability` → Simulates dropped heatmap packets
- Each drone:
  1. Runs segmentation on the same image (with small perturbations for realism)
  2. Compresses the mask into a **heatmap** (grid of weed densities 0..1)
  3. Simulates network transmission (may be dropped based on probability)
- A simple **animation** shows drones sending packets to a central server.

**Example Drone Metrics Output:**

| Metric | Value |
|--------|-------|
| Packets Sent | 6 |
| Packets Received | 6 |
| Original Size | 2522.11 KB |
| Compressed Size | 18.75 KB |
| Bandwidth Saved | 99.3% |
| Success Rate | 100.0% |

**Explanation:**  
- Packets sent = total drones simulated.  
- Packets received = successfully transmitted heatmaps.  
- Bandwidth saved shows **compression efficiency**.  
- Success rate shows **network reliability**.

---

## 4️⃣ Heatmap Generation

- Each drone generates a **compressed heatmap** from the mask using `mask_to_heatmap`.
- Heatmaps represent **weed density per grid cell**.
- All valid heatmaps are aggregated to produce a **farm-wide weed map**.
- Aggregated map helps in **planning herbicide spraying**.

---

## 4.5 Transmission Testing

### HTTP (Flask server)

1. Start the Flask server:

```bash
python src/server.py
```

# 5. Installation & Usage

###  1. Clone repository:

```bash
git clone <repo-url>
cd WeedIoT
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Start Flask server:

```bash
python src/server.py 
```
or 
```bash
python3 -m src.server
```

### 4. Run Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

---

### 5. Upload crop images and configure drone simulation.

---

# Handling Mosquitto MQTT

If you're using MQTT to handle the communication between drones and the central server, you will need to interact with **Mosquitto**, the MQTT broker. Below are the instructions to start and stop Mosquitto on different platforms.

---

## Starting Mosquitto

### On Linux:
To start the Mosquitto service, use the following command:

```bash
sudo systemctl start mosquitto
```

### On Windows:
If Mosquitto is running as a service on Windows, you can start it by running this command in the Command Prompt:

```bash
net start mosquitto
```

---

## Stopping Mosquitto

### On Linux:
To stop the Mosquitto service, run the following command:

```bash
sudo systemctl stop mosquitto
```

### On Windows:
If Mosquitto is running as a service on Windows, you can stop it by running this command in the Command Prompt:

```bash
net stop mosquitto
```

---

These commands allow you to control the Mosquitto MQTT broker in your project, enabling communication between drones and the central server.
