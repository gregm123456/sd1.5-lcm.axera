# Parallel Service Experiment Results

This document summarizes the experiments conducted to test the scalability of the Pi Axera SD Generator service on the AX650N hardware.

## Overview
We conducted load tests to determine if running multiple instances of the SD service in parallel improves total image generation throughput. We tested 1, 2, and 3 concurrent instances.

## Scripts
*   `run_parallel_experiment.py`: Automatically launches additional temporary instances (ports 5001, 5002), waits for them to initialize, runs a parallel load test against all active instances, and then shuts them down.
*   `run_single_instance_experiment.py`: Runs a baseline sequential load test against the main service instance on port 5000.

## Results Summary

| Configuration | Latency (per image) | Total Throughput | Notes |
| :--- | :--- | :--- | :--- |
| **1 Instance** | **~3.00s** | 0.33 imgs/sec | **Optimal Configuration.** Best user experience. |
| **2 Instances** | ~5.20s | **0.38 imgs/sec** | Latency increased by 73% for only 15% throughput gain. |
| **3 Instances** | ~7.80s | **0.38 imgs/sec** | Latency increased by 160% with **no additional throughput gain** vs 2 instances. |
| **4 Instances** | N/A | N/A | **Failed to start.** Hardware limit reached. |

## Key Findings

1.  **Hardware Limit:** The AX650N NPU supports a maximum of **3 concurrent model instances**. Attempting to load a 4th instance fails with `axclrtEngineLoadFromFile failed`, likely due to NPU context or memory handle limits.
2.  **Throughput Ceiling:** The total system throughput hits a hard ceiling at **~0.38 images/second**. The NPU is fully saturated at this point.
3.  **Diminishing Returns:** Adding parallel instances causes significant resource contention. The NPU handles serialized requests (one after another) much more efficiently than parallel requests (context switching between models).
    *   Going from 1 to 2 instances increases wait time from 3s to 5.2s.
    *   Going from 2 to 3 instances increases wait time from 5.2s to 7.8s.

## Recommendation
**Stick to a single instance.**
The small gain in total throughput (0.05 imgs/sec) does not justify the massive degradation in individual request latency. Users will prefer waiting 3 seconds per image rather than 5-8 seconds, and the total volume of images produced per minute is nearly the same.

## How to Reproduce

### 1. Single Instance Baseline
Ensure the main service is running on port 5000.
```bash
python3 run_single_instance_experiment.py
```

### 2. Parallel Test
This script will launch temporary instances on ports 5001 and 5002, run the test, and clean them up.
```bash
python3 run_parallel_experiment.py
```

---

## Appendix: Systemd Configuration (Reference)

If you still wish to run multiple instances permanently (e.g., for redundancy rather than throughput), you can use a parameterized systemd service.

1.  Create a file `/etc/systemd/system/pi_axera_sd_generator@.service`:

    ```ini
    [Unit]
    Description=Pi Axera SD Image Generator Service (Port %i)
    After=network.target

    [Service]
    Type=simple
    User=gregm
    WorkingDirectory=/home/gregm/sd1.5-lcm.axera_greg
    ExecStart=/home/gregm/sd1.5-lcm.axera_greg/.venv/bin/python pi_axera_sd_service/pi_axera_sd_generator.py
    Restart=on-failure
    Environment=VIRTUAL_ENV=/home/gregm/sd1.5-lcm.axera_greg/.venv
    Environment=PATH=/home/gregm/sd1.5-lcm.axera_greg/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    Environment=TEXT_MODEL_DIR=/home/gregm/sd1.5-lcm.axera_greg/models/
    Environment=TEXT_ENCODER_MODEL=/home/gregm/sd1.5-lcm.axera_greg/models/rv_optimized/sd15_text_encoder_sim.axmodel
    Environment=UNET_MODEL=/home/gregm/sd1.5-lcm.axera_greg/models/rv_optimized/unet.axmodel
    Environment=VAE_DECODER_MODEL=/home/gregm/sd1.5-lcm.axera_greg/models/rv_optimized/vae_decoder.axmodel
    Environment=VAE_ENCODER_MODEL=/home/gregm/sd1.5-lcm.axera_greg/models/rv_optimized/vae_encoder.axmodel
    Environment=TIME_INPUT_TXT2IMG=/home/gregm/sd1.5-lcm.axera_greg/models/time_input_txt2img.npy
    Environment=TIME_INPUT_IMG2IMG=/home/gregm/sd1.5-lcm.axera_greg/models/time_input_img2img.npy
    Environment=OUTPUT_DIR=/home/gregm/sd1.5-lcm.axera_greg/pi_axera_sd_service/output
    Environment=HOST=0.0.0.0
    Environment=PORT=%i

    [Install]
    WantedBy=multi-user.target
    ```

2.  Start instances:
    ```bash
    sudo systemctl enable --now pi_axera_sd_generator@5000
    sudo systemctl enable --now pi_axera_sd_generator@5001
    ```
