import requests
import base64

FILENAME = "test.wav"

with open(FILENAME, mode="rb") as f:
    audio_encoded = base64.b64encode(f.read())

encoded_string = "data:audio/wav;base64," + str(audio_encoded, 'utf-8')

r = requests.post(
    url='http://localhost:8080/api/predict/', 
    json={
        "data": [
            {"data": encoded_string, "name": FILENAME}]})
r.json()

print(r.json())