# Step 5: Model Compilation on Intel Mac

This guide provides standalone instructions for compiling the Realistic Vision V6.0 B1 models for the Axera AX650N using an Intel Mac. This offloads the heavy quantization and compilation work from the Raspberry Pi.

## 1. Transfer and Organize Files on Mac

To avoid duplicating large files on the Raspberry Pi, you will copy the necessary components in batches and organize them on your Intel Mac.

### A. Files to Copy from Raspberry Pi to Mac
Run these commands **on the Raspberry Pi** to push the files to your Mac. This avoids tarballs and copies only the necessary files.

```bash
# 1. Create the destination folder on the Mac and clean up any partial venv
ssh gregm2@192.168.4.130 "mkdir -p ~/pulsar_build/model_realisticvision && rm -rf ~/pulsar_build/model_realisticvision/venv"

# 2. Copy the main work folder (EXCLUDING venv)
# We copy subdirectories individually to avoid the large, platform-specific venv
scp -r /home/gregm/sd1.5-lcm.axera_greg/model_realisticvision/onnx-models gregm2@192.168.4.130:~/pulsar_build/model_realisticvision/
scp -r /home/gregm/sd1.5-lcm.axera_greg/model_realisticvision/datasets gregm2@192.168.4.130:~/pulsar_build/model_realisticvision/
scp -r /home/gregm/sd1.5-lcm.axera_greg/model_realisticvision/5.1 gregm2@192.168.4.130:~/pulsar_build/model_realisticvision/
scp /home/gregm/sd1.5-lcm.axera_greg/model_realisticvision/*.py gregm2@192.168.4.130:~/pulsar_build/model_realisticvision/
scp /home/gregm/sd1.5-lcm.axera_greg/model_realisticvision/*.md gregm2@192.168.4.130:~/pulsar_build/model_realisticvision/

# 3. Copy shared config files (individual files to avoid symlink issues)
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/configs/text_encoder_u16.json gregm2@192.168.4.130:~/pulsar_build/
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/configs/unet_u16.json gregm2@192.168.4.130:~/pulsar_build/
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/configs/vae_encoder_u16.json gregm2@192.168.4.130:~/pulsar_build/
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/configs/vae_u16.json gregm2@192.168.4.130:~/pulsar_build/

# 4. Copy shared tokenizer files (individual files to avoid symlink issues)
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/tokenizer/merges.txt gregm2@192.168.4.130:~/pulsar_build/
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/tokenizer/special_tokens_map.json gregm2@192.168.4.130:~/pulsar_build/
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/tokenizer/tokenizer_config.json gregm2@192.168.4.130:~/pulsar_build/
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/tokenizer/vocab.json gregm2@192.168.4.130:~/pulsar_build/

# 5. Copy shared calibration datasets
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/datasets/text_encoder_calibration.tar gregm2@192.168.4.130:~/pulsar_build/
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/datasets/imagenet-calib.tar gregm2@192.168.4.130:~/pulsar_build/
```

### B. Organize the Structure on the Mac
Once the files are on your Mac, organize them so the Pulsar2 commands can find everything. Navigate to your `model_realisticvision` folder on the Mac and run:

```bash
# 1. Remove the broken symlinks from the Pi
rm configs tokenizer

# 2. Create the configs and tokenizer directories and move files
mkdir configs
mkdir tokenizer

# Move the config files
mv ../text_encoder_u16.json ./configs/
mv ../unet_u16.json ./configs/
mv ../vae_encoder_u16.json ./configs/
mv ../vae_u16.json ./configs/

# Move the tokenizer files
mv ../merges.txt ./tokenizer/
mv ../special_tokens_map.json ./tokenizer/
mv ../tokenizer_config.json ./tokenizer/
mv ../vocab.json ./tokenizer/

# 3. Move the shared calibration files into the datasets folder
mv ../text_encoder_calibration.tar ./datasets/
mv ../imagenet-calib.tar ./datasets/
```

### Final Structure Check (On Mac)
Your `model_realisticvision` folder on the Mac should look like this:
*   `onnx-models/`
*   `datasets/` (Contains `calib_data_unet/`, `calib_data_vae/`, `text_encoder_calibration.tar`, `imagenet-calib.tar`)
*   `configs/` (Actual folder, not a symlink)
*   `tokenizer/` (Actual folder, not a symlink)
*   `ax_pulsar2_5.1.tar.gz` (The image file)

## 2. Prerequisites on Intel Mac

1.  **Install Docker Desktop**: Download and install from [docker.com](https://www.docker.com/products/docker-desktop/).
2.  **Allocate Resources**: In Docker Desktop settings, ensure you have allocated at least **16GB of RAM** (U-Net compilation is memory-intensive).
3.  **Load the Pulsar2 Toolchain**:
    If you haven't already, download the Pulsar2 Docker image (v5.1) from Hugging Face:
    [AXERA-TECH/Pulsar2/5.1/ax_pulsar2_5.1.tar.gz](https://huggingface.co/AXERA-TECH/Pulsar2/blob/main/5.1/ax_pulsar2_5.1.tar.gz)

    Open a terminal on your Mac, navigate to the folder containing the `.tar.gz` file, and run:
    ```bash
    docker load -i ax_pulsar2_5.1.tar.gz
    ```

    Verify it is loaded:
    ```bash
    docker images | grep pulsar2
    ```

## 3. Run Compilation Commands

Run these commands from within the `model_realisticvision` directory on your Mac. We will mount the current directory to `/data` inside the container.

### A. Text Encoder
```bash
docker run --rm -v "$(pwd)":/data -w /data pulsar2:5.1 \
    pulsar2 build --input onnx-models/sd15_text_encoder_sim.onnx \
    --output_dir axmodels --output_name sd15_text_encoder_sim.axmodel \
    --config configs/text_encoder_u16.json
```

### B. VAE Encoder
```bash
docker run --rm -v "$(pwd)":/data -w /data pulsar2:5.1 \
    pulsar2 build --input onnx-models/vae_encoder.onnx \
    --output_dir axmodels --output_name vae_encoder.axmodel \
    --config configs/vae_encoder_u16.json
```

### C. VAE Decoder
```bash
docker run --rm -v "$(pwd)":/data -w /data pulsar2:5.1 \
    pulsar2 build --input onnx-models/vae_decoder.onnx \
    --output_dir axmodels --output_name vae_decoder.axmodel \
    --config configs/vae_u16.json
```

### D. U-Net (Time Consuming)
```bash
docker run --rm -v "$(pwd)":/data -w /data pulsar2:5.1 \
    pulsar2 build --input onnx-models/unet.onnx \
    --output_dir axmodels --output_name unet.axmodel \
    --config configs/unet_u16.json
```

## 4. Next Steps: Back to Raspberry Pi

Once the commands finish, you will have a new `axmodels/` directory containing four `.axmodel` files.

1.  **Transfer Files**: Copy the contents of the `axmodels/` folder back to the Raspberry Pi, specifically into the `/home/gregm/sd1.5-lcm.axera_greg/models/` directory.
2.  **Verify Files**: Ensure the following files are present on the Pi:
    *   `models/sd15_text_encoder_sim.axmodel`
    *   `models/vae_encoder.axmodel`
    *   `models/vae_decoder.axmodel`
    *   `models/unet.axmodel`
3.  **Resume Plan**: Continue with **Step 6 (Deploy)** and **Step 7 (Test Inference)** in the main `model_conversion_plan.md` on the Raspberry Pi.

---
**References:**
- [Pulsar2 Documentation](https://github.com/gregm123456/pulsar2-docs)
- [Axera Hugging Face](https://huggingface.co/AXERA-TECH)
