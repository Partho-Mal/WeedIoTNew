"The HTTP server receives JSON payloads containing the drone’s compressed heatmaps. Each POST request updates the farm-wide aggregated map. We tested it locally using Flask and verified that each simulated drone packet is successfully received."


“The MQTT broker receives the compressed heatmaps from each drone as JSON messages on the topic farm/weed_data. We subscribed to the topic and verified that messages are correctly received and contain the expected data.”



3️⃣ Optional Notes

HTTP is request/response, easier for testing.
MQTT is publish/subscribe, better for real IoT networks.
You can demonstrate both by showing server console logs or mosquitto_sub output.