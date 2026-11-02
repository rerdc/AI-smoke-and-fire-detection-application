import json
import paho.mqtt.client as mqtt

client = mqtt.Client()


def connect_mqtt(host="localhost", port=1883):
    client.connect(host, port, 60)
    client.loop_start()


def send_mqtt(warning_level, smoke_conf, fire_prob):
    payload = {
        "warning_level": int(warning_level),
        "smoke_conf": float(smoke_conf),
        "fire_prob": float(fire_prob),
    }

    client.publish(
        "fire_detection/status",
        json.dumps(payload),
    )


def disconnect_mqtt():
    client.loop_stop()
    client.disconnect()
