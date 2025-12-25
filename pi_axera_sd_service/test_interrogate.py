import requests
import base64
import os

def test_interrogate():
    url = "http://127.0.0.1:5000/sdapi/v1/interrogate"
    images = [
        "/home/gregm/sd1.5-lcm.axera_greg/pi_axera_sd_service/Cadillac-420x279.png",
        "/home/gregm/sd1.5-lcm.axera_greg/pi_axera_sd_service/output/gen_1766556888344.png",
        "/home/gregm/sd1.5-lcm.axera_greg/pi_axera_sd_service/output/gen_1766683269858.png"
    ]
    
    for image_path in images:
        if not os.path.exists(image_path):
            print(f"Error: {image_path} not found")
            continue

        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')

        payload = {
            "image": img_base64
        }

        print(f"Sending interrogation request for {os.path.basename(image_path)}...")
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("Success!")
            print("Caption:", data.get("caption"))
            print("\nDetailed Scores:")
            for item in data.get("interrogations", []):
                print(f"  {item['tag']:<25} : {item['score']:.4f}")
        else:
            print(f"Error {response.status_code}: {response.text}")
        print("-" * 20)

if __name__ == "__main__":
    test_interrogate()
