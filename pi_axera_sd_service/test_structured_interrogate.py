import requests
import base64
import json
import os

def test_structured_interrogate():
    url = "http://127.0.0.1:5000/sdapi/v1/interrogate/structured"
    image_path = "/home/gregm/sd1.5-lcm.axera_greg/assets/img2img-init.png"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image": image_b64,
        "categories": {
            "subject": ["person", "animal", "vehicle", "landscape", "building"],
            "gender": ["man", "woman", "boy", "girl"],
            "hair_color": ["pink", "blonde", "brown", "black", "red"],
            "setting": ["indoor", "outdoor", "studio", "nature"]
        }
    }

    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Response received:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")

if __name__ == "__main__":
    test_structured_interrogate()
