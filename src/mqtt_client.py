# src/mqtt_client.py
import json
import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "farm/weed_data"

def send_heatmap_mqtt(heatmap, drone_id="drone_1"):
    payload = json.dumps({"drone_id": drone_id, "heatmap": heatmap.tolist()})
    client = mqtt.Client()
    try:
        client.connect(BROKER_HOST, BROKER_PORT, 60)
        client.publish(TOPIC, payload)
        client.disconnect()
        return True
    except Exception as e:
        print("MQTT send failed:", e)
        return False
