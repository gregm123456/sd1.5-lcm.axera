## Realistic Vision V6.0 B1 Conversion Plan for Axera AX650N / LLM8850

This plan details the deterministic steps to download, convert, and fuse Realistic Vision V6.0 B1 (with VAE and LCM-LoRA) for use on the Axera AX650N (NPU3) platform.
All work (downloads, outputs, and intermediate files) will be done inside the `model_realisticvision` directory.

### 1. Environment Preparation
- Ensure you have the Axera Pulsar2 toolchain Docker image and Python 3.8+.
- **Use a Virtual Environment**:
	```bash
	cd model_realisticvision
	python3 -m venv venv
	source venv/bin/activate
	pip install -r ../model_convert/requirements.txt
	pip install loguru onnxscript  # Additional dependencies found during execution
	```
- **Setup Workspace Links**: Symlink the shared configs and tokenizer to this directory so scripts can find them locally.
	```bash
	ln -s ../model_convert/configs ./configs
	ln -s ../model_convert/tokenizer ./tokenizer
	```

### 2. Download Model Files

#### a. Download Realistic Vision V6.0 B1 (noVAE) from Hugging Face
	```bash
	huggingface-cli download SG161222/Realistic_Vision_V6.0_B1_noVAE --local-dir ./hugging_face/Realistic_Vision_V6.0_B1
	```
	*Note: If `vocab.json` fails consistency check (common with this model), force download it specifically:*
	```bash
	huggingface-cli download SG161222/Realistic_Vision_V6.0_B1_noVAE tokenizer/vocab.json --local-dir ./hugging_face/Realistic_Vision_V6.0_B1 --force-download
	```

#### b. Download the recommended VAE (Diffusers compatible)
	```bash
	huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir ./hugging_face/vae-ft-mse
	```
#### c. Replace the VAE in the base model directory
	```bash
	rm -rf ./hugging_face/Realistic_Vision_V6.0_B1/vae
	cp -r ./hugging_face/vae-ft-mse ./hugging_face/Realistic_Vision_V6.0_B1/vae
	```
#### d. Download LCM-LoRA weights
	```bash
	huggingface-cli download latent-consistency/lcm-lora-sdv1-5 --local-dir ./lcm-lora-sdv1-5
	```

### 3. Export to ONNX with LoRA Fusion

To avoid Out-Of-Memory (OOM) issues on systems with limited RAM (e.g., 16GB), components are exported individually using `sd15_export_onnx.py`.

```bash
# 1. Export Text Encoder
python3 ../model_convert/sd15_export_onnx.py --input_path ./hugging_face/Realistic_Vision_V6.0_B1 --input_lora_path ./lcm-lora-sdv1-5/ --output_path ./onnx-models --only_text

# 2. Export VAE (Encoder & Decoder)
python3 ../model_convert/sd15_export_onnx.py --input_path ./hugging_face/Realistic_Vision_V6.0_B1 --input_lora_path ./lcm-lora-sdv1-5/ --output_path ./onnx-models --only_vae

# 3. Export U-Net (Fused with LCM-LoRA)
# Note: We skip onnxsim here to prevent OOM during simplification.
python3 ../model_convert/sd15_export_onnx.py --input_path ./hugging_face/Realistic_Vision_V6.0_B1 --input_lora_path ./lcm-lora-sdv1-5/ --output_path ./onnx-models --only_unet
```

**Key Export Settings:**
- `opset_version=18`: Required for compatibility with modern PyTorch exports.
- `torch.no_grad()` and `gc.collect()`: Used to manage memory during export.

#### U-Net Graph Surgery (Required for Axera Pipeline)
The Axera inference pipeline expects the U-Net to have a direct input for the 1280-dim time embedding, rather than calculating it from a scalar `t`.

1.  **Run Graph Surgery**: Use `final_fix_unet.py` to remove the internal time projection nodes and expose the input `/down_blocks.0/resnets.0/act_1/Mul_output_0`.
    ```bash
    python3 final_fix_unet.py
    mv onnx-models/unet_fixed.onnx onnx-models/unet.onnx
    ```
2.  **Cleanup Duplicates**: Ensure no duplicate input definitions exist.
    ```bash
    python3 fix_duplicates.py
    mv onnx-models/unet_fixed.onnx onnx-models/unet.onnx
    ```

### 4. Generate Calibration Data

Quantization requires a representative dataset to determine optimal scaling factors.
```bash
python3 ../model_convert/sd15_lora_prepare_data.py --export_onnx_dir ./onnx-models
```
This generates `data.tar` files in `datasets/calib_data_unet/` and `datasets/calib_data_vae/`.

### 5. Compile ONNX to AXMODEL (Pulsar2)

**Note: Compilation is extremely resource-intensive. It is highly recommended to offload this step to an Intel Mac as detailed in [step5_intel_mac_final.md](step5_intel_mac_final.md).**

#### a. Prepare Calibration Data (Critical)
Pulsar2 requires `float32` calibration data. The VAE dataset often contains `float64` which must be converted:
1. Extract `datasets/calib_data_vae/data.tar`.
2. Convert all `.npy` files from `float64` to `float32`.
3. Re-pack the `.tar` file.

#### b. Run Compilation
Run these commands inside the Pulsar2 toolchain Docker environment. These commands perform quantization and precision analysis.

```bash
# Text Encoder
pulsar2 build --input onnx-models/sd15_text_encoder_sim.onnx --output_dir axmodels --output_name sd15_text_encoder_sim.axmodel --config configs/text_encoder_u16.json

# VAE Encoder
pulsar2 build --input onnx-models/vae_encoder.onnx --output_dir axmodels --output_name vae_encoder.axmodel --config configs/vae_encoder_u16.json

# VAE Decoder
pulsar2 build --input onnx-models/vae_decoder.onnx --output_dir axmodels --output_name vae_decoder.axmodel --config configs/vae_u16.json

# U-Net (This step is time-consuming)
pulsar2 build --input onnx-models/unet.onnx --output_dir axmodels --output_name unet.axmodel --config configs/unet_u16.json
```


### 6. Deploy Compiled Models

To avoid overwriting the base SD1.5 models and to maintain compatibility with the `pi_axera_sd_generator` service, deploy the models to a dedicated subdirectory:

1. **Create Directory**: `mkdir -p ../models/rv_optimized`
2. **Copy Models**: 
   ```bash
   cp axmodels/*.axmodel ../models/rv_optimized/
   ```
3. **Verify Service Config**: Ensure `pi_axera_sd_generator.service` is configured to use the `rv_optimized` paths for `UNET_MODEL`, `VAE_DECODER_MODEL`, and `TEXT_ENCODER_MODEL`.

### 7. Test Inference
Run a test using the provided Python scripts (e.g., `run_txt2img_axe_infer.py`) or start the service to verify correct operation.

---
**Notes:**
- **Memory Management**: Exporting the U-Net requires significant RAM. If the process crashes, ensure no other heavy processes are running and that `onnxsim` is disabled for the U-Net export.
- **Graph Surgery**: The U-Net input name `/down_blocks.0/resnets.0/act_1/Mul_output_0` is hardcoded in the Axera inference scripts. If the graph structure changes in future SD versions, this name may need to be updated in both the surgery script and the inference code.
- **Data Types**: Ensure all inputs to ONNX Runtime during calibration are explicitly cast to `np.float32`. Using `double` (default for some numpy operations) will cause type mismatch errors.
- **Reproducibility**: Always use the same Hugging Face and LoRA versions. For other SD1.5-based models, repeat the above steps, adjusting only the base model path.
