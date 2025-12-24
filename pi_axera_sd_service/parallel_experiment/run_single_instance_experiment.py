import time
import json
import urllib.request
import urllib.error

# Configuration
PORT = 5000
REQUESTS = 10
PROMPT = "a beautiful landscape with mountains and a lake, realistic, 8k"

def send_request(request_id):
    url = f"http://127.0.0.1:{PORT}/sdapi/v1/txt2img"
    data = json.dumps({"prompt": PROMPT}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            resp_data = json.load(response)
            duration = time.time() - start
            
            gen_time = "N/A"
            if "info" in resp_data:
                gen_time = resp_data["info"]
            elif "total_time_ms" in resp_data:
                gen_time = f"{resp_data['total_time_ms']}ms"
                
            print(f"Req {request_id}: Success in {duration:.2f}s ({gen_time})")
            return duration
    except Exception as e:
        print(f"Req {request_id}: Failed - {e}")
        return None

def main():
    print(f"Starting single instance load test on port {PORT}...")
    print(f"Sending {REQUESTS} sequential requests...")
    
    times = []
    start_time = time.time()
    
    for i in range(REQUESTS):
        t = send_request(i+1)
        if t is not None:
            times.append(t)
            
    total_duration = time.time() - start_time
    
    if times:
        avg = sum(times) / len(times)
        print(f"\n--- Finished ---")
        print(f"Total Time: {total_duration:.2f}s")
        print(f"Average Latency: {avg:.2f}s")
        print(f"Throughput: {len(times)/total_duration:.2f} imgs/sec")
    else:
        print("All requests failed.")

if __name__ == "__main__":
    main()
