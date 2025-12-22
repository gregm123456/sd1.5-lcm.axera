# Step 5: Model Compilation on Intel Mac

This guide provides standalone instructions for compiling the Realistic Vision V6.0 B1 models for the Axera AX650N using an Intel Mac. This offloads the heavy quantization and compilation work from the Raspberry Pi.

## 1. Transfer and Organize Files on Mac

### A. Files to Copy from Raspberry Pi to Mac
Run these commands **on the Raspberry Pi** to push the files to your Mac.

```bash
# 1. Create the destination folder on the Mac
ssh gregm2@192.168.4.130 "mkdir -p ~/pulsar_build/model_realisticvision"

# 2. Copy the main work folder components
scp -r /home/gregm/sd1.5-lcm.axera_greg/model_realisticvision/onnx-models gregm2@192.168.4.130:~/pulsar_build/model_realisticvision/
scp -r /home/gregm/sd1.5-lcm.axera_greg/model_realisticvision/datasets gregm2@192.168.4.130:~/pulsar_build/model_realisticvision/
scp /home/gregm/sd1.5-lcm.axera_greg/model_realisticvision/*.py gregm2@192.168.4.130:~/pulsar_build/model_realisticvision/
scp /home/gregm/sd1.5-lcm.axera_greg/model_realisticvision/*.md gregm2@192.168.4.130:~/pulsar_build/model_realisticvision/

# 3. Copy shared config files to the root of the build folder
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/configs/text_encoder_u16.json gregm2@192.168.4.130:~/pulsar_build/
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/configs/unet_u16.json gregm2@192.168.4.130:~/pulsar_build/
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/configs/vae_encoder_u16.json gregm2@192.168.4.130:~/pulsar_build/
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/configs/vae_u16.json gregm2@192.168.4.130:~/pulsar_build/

# 4. Copy shared calibration datasets to the root
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/datasets/text_encoder_calibration.tar gregm2@192.168.4.130:~/pulsar_build/
scp /home/gregm/sd1.5-lcm.axera_greg/model_convert/datasets/imagenet-calib.tar gregm2@192.168.4.130:~/pulsar_build/
```

### B. Organize the Structure on the Mac
Navigate to `~/pulsar_build/model_realisticvision` on your Mac and run:

```bash
# 1. Create the configs directory
mkdir -p configs

# 2. Move the config files from the parent directory
mv ../text_encoder_u16.json ./configs/
mv ../unet_u16.json ./configs/
mv ../vae_encoder_u16.json ./configs/
mv ../vae_u16.json ./configs/

# 3. Move the shared calibration files into the datasets folder
mv ../text_encoder_calibration.tar ./datasets/
mv ../imagenet-calib.tar ./datasets/
```

## 2. Prepare Calibration Data (Critical)

The VAE calibration data often contains `float64` arrays, but Pulsar2 requires `float32`. You must convert them on your Mac before compiling.

1.  **Install Requirements**:
    ```bash
    pip3 install numpy
    ```

2.  **Run the Fix Script**:
    Create a file named `fix_calib.py` in `model_realisticvision/` and run it:
    ```python
    import numpy as np
    import os
    import tarfile

    def fix_tar(tar_path):
        extract_path = tar_path + "_extracted"
        os.makedirs(extract_path, exist_ok=True)
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=extract_path)
        
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.npy'):
                    p = os.path.join(root, file)
                    data = np.load(p, allow_pickle=True)
                    if data.dtype == np.object_:
                        obj = data.item()
                        for k, v in obj.items():
                            if hasattr(v, 'dtype') and v.dtype == np.float64:
                                obj[k] = v.astype(np.float32)
                        np.save(p, obj)
                    elif data.dtype == np.float64:
                        np.save(p, data.astype(np.float32))
        
        with tarfile.open(tar_path, 'w') as tar:
            tar.add(extract_path, arcname='.')
    
    fix_tar('datasets/calib_data_vae/data.tar')
    ```
    ```bash
    python3 fix_calib.py
    ```

## 3. Prerequisites on Intel Mac

1.  **Docker Desktop**: Ensure it is running.
2.  **Resources**: Allocate at least **16GB of RAM** in Docker Desktop settings.
3.  **Load Image**:
    ```bash
    docker load -i ax_pulsar2_5.1.tar.gz
    ```

## 4. Run Compilation Commands

Run these from the `model_realisticvision` directory.

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

## 5. Next Steps: Back to Raspberry Pi

1.  **Transfer**: Copy the compiled models back to the Raspberry Pi. Run this from your Mac:
    ```bash
    scp axmodels/*.axmodel gregm@192.168.4.121:/home/gregm/sd1.5-lcm.axera_greg/models/rv_optimized/
    ```
2.  **Verify**: Ensure the following files are present on the Pi in the `models/rv_optimized/` directory:
    *   `sd15_text_encoder_sim.axmodel`
    *   `vae_encoder.axmodel`
    *   `vae_decoder.axmodel`
    *   `unet.axmodel`
3.  **Resume**: Continue with **Step 6** in `model_conversion_plan.md` on the Raspberry Pi.

---
**References:**
- [Pulsar2 Documentation](https://github.com/gregm123456/pulsar2-docs)
- [Axera Hugging Face](https://huggingface.co/AXERA-TECH)
