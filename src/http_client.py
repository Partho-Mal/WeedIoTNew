# src/http_client.py
import requests
import json

SERVER_URL = "http://127.0.0.1:5000/upload"

def send_heatmap_http(heatmap, drone_id="drone_1"):
    payload = {"drone_id": drone_id, "heatmap": heatmap.tolist()}
    try:
        r = requests.post(SERVER_URL, json=payload, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print("HTTP send failed:", e)
        return False
