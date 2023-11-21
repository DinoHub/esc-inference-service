"""
Test for running a single inference via gradio API.
"""

import base64
import requests

FILENAME = "../examples/busy_street_cook_16k.wav"

with open(FILENAME, mode="rb") as f:
    audio_encoded = base64.b64encode(f.read())

encoded_string = f"data:audio/wav;base64,{str(audio_encoded, 'utf-8')}"

r = requests.post(
    url="http://localhost:8080/api/predict/",
    json={"data": [{"data": encoded_string, "name": FILENAME}]},
    timeout=30,
)
r.json()

print(r.json())
