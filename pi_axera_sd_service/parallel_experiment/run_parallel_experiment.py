import os
import subprocess
import time
import json
import threading
import urllib.request
import urllib.error
import signal
import sys
from datetime import datetime

# Configuration
BASE_DIR = "/home/gregm/sd1.5-lcm.axera_greg"
PYTHON_EXEC = os.path.join(BASE_DIR, ".venv/bin/python")
SCRIPT_PATH = os.path.join(BASE_DIR, "pi_axera_sd_service/pi_axera_sd_generator.py")
ENV_VARS = {
    "VIRTUAL_ENV": os.path.join(BASE_DIR, ".venv"),
    "PATH": f"{os.path.join(BASE_DIR, '.venv/bin')}:{os.environ.get('PATH', '')}",
    "TEXT_MODEL_DIR": os.path.join(BASE_DIR, "models/"),
    "TEXT_ENCODER_MODEL": os.path.join(BASE_DIR, "models/rv_optimized/sd15_text_encoder_sim.axmodel"),
    "UNET_MODEL": os.path.join(BASE_DIR, "models/rv_optimized/unet.axmodel"),
    "VAE_DECODER_MODEL": os.path.join(BASE_DIR, "models/rv_optimized/vae_decoder.axmodel"),
    "VAE_ENCODER_MODEL": os.path.join(BASE_DIR, "models/rv_optimized/vae_encoder.axmodel"),
    "TIME_INPUT_TXT2IMG": os.path.join(BASE_DIR, "models/time_input_txt2img.npy"),
    "TIME_INPUT_IMG2IMG": os.path.join(BASE_DIR, "models/time_input_img2img.npy"),
    "OUTPUT_DIR": os.path.join(BASE_DIR, "pi_axera_sd_service/output"),
    "HOST": "0.0.0.0"
}

EXTRA_PORTS = [5001]
ALL_PORTS = [5000] + EXTRA_PORTS
REQUESTS_PER_INSTANCE = 10
PROMPT = "a beautiful landscape with mountains and a lake, realistic, 8k"

processes = []

def start_instance(port):
    env = os.environ.copy()
    env.update(ENV_VARS)
    env["PORT"] = str(port)
    
    # Use a separate output directory for each instance
    instance_output_dir = f"{ENV_VARS['OUTPUT_DIR']}_{port}"
    os.makedirs(instance_output_dir, exist_ok=True)
    env["OUTPUT_DIR"] = instance_output_dir
    
    print(f"Starting instance on port {port} (Output: {instance_output_dir})...")
    # Redirect stdout/stderr to avoid cluttering the console, or log to file
    log_file = open(f"instance_{port}.log", "w")
    p = subprocess.Popen(
        [PYTHON_EXEC, SCRIPT_PATH],
        env=env,
        cwd=BASE_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    return p, log_file

def wait_for_service(port, timeout=60):
    url = f"http://127.0.0.1:{port}/" # Root might not return 200, but connection refused means not ready
    # Actually the service doesn't seem to have a root route, but we can try connecting.
    # Or try a known route like /sdapi/v1/txt2img with invalid method to get 405 or something.
    # Let's just try to connect.
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Just check if we can connect
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as response:
                return True
        except urllib.error.HTTPError:
            # 404 is fine, means server is running
            return True
        except urllib.error.URLError:
            time.sleep(1)
            continue
        except Exception:
            time.sleep(1)
            continue
    return False

def send_request(port, request_id):
    url = f"http://127.0.0.1:{port}/sdapi/v1/txt2img"
    data = json.dumps({"prompt": PROMPT}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            resp_data = json.load(response)
            duration = time.time() - start
            # Try to get generation time from response if available
            gen_time = "N/A"
            if "info" in resp_data:
                gen_time = resp_data["info"]
            elif "total_time_ms" in resp_data:
                gen_time = f"{resp_data['total_time_ms']}ms"
                
            print(f"[Port {port}] Req {request_id}: Success in {duration:.2f}s ({gen_time})")
            return duration
    except Exception as e:
        print(f"[Port {port}] Req {request_id}: Failed - {e}")
        return None

def worker(port):
    print(f"Starting load test on port {port}...")
    times = []
    for i in range(REQUESTS_PER_INSTANCE):
        t = send_request(port, i+1)
        if t is not None:
            times.append(t)
    
    if times:
        avg = sum(times) / len(times)
        print(f"--- Port {port} Finished. Avg: {avg:.2f}s, Total: {sum(times):.2f}s ---")
    else:
        print(f"--- Port {port} Finished. All failed. ---")

def cleanup(signum=None, frame=None):
    print("\nCleaning up...")
    for p, f in processes:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
        f.close()
    print("Done.")
    if signum:
        sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

def main():
    # 1. Start extra instances
    for port in EXTRA_PORTS:
        p, f = start_instance(port)
        processes.append((p, f))
    
    # 2. Wait for them to be ready
    print("Waiting for services to start...")
    # Give them a head start
    time.sleep(5) 
    
    # Check port 5000 too
    if not wait_for_service(5000, timeout=5):
        print("Warning: Port 5000 (main service) does not seem to be responding. Is it running?")
    
    active_ports = [5000]
    
    for port in EXTRA_PORTS:
        if wait_for_service(port):
            print(f"Service on port {port} is ready.")
            active_ports.append(port)
        else:
            print(f"Service on port {port} failed to start.")
            # Print tail of log
            log_filename = f"instance_{port}.log"
            if os.path.exists(log_filename):
                print(f"--- Tail of {log_filename} ---")
                try:
                    with open(log_filename, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-10:]:
                            print(line.strip())
                except Exception as e:
                    print(f"Could not read log: {e}")
                print("-------------------------------")
            
            # Do not abort, just continue with what we have
            # cleanup()
            # return

    if len(active_ports) <= 1:
        print("Not enough services running to perform a meaningful parallel test.")
        cleanup()
        return

    print(f"\nStarting parallel load test with {len(active_ports)} instances: {active_ports}")
    print(f"Sending {REQUESTS_PER_INSTANCE} requests to each instance.")
    
    threads = []
    start_time = time.time()
    
    for port in active_ports:
        t = threading.Thread(target=worker, args=(port,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    total_duration = time.time() - start_time
    print(f"\nTest completed in {total_duration:.2f}s")
    
    cleanup()

if __name__ == "__main__":
    main()
