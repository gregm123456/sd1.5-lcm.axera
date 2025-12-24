# Hugging Face Repository Plan: Realistic Vision V6.0 B1 (Axera Optimized)

This document outlines the plan for creating and publishing a Hugging Face model repository for the Axera-optimized version of Realistic Vision V6.0 B1.

## 1. Repository Identity
- **Proposed Name**: `realistic-vision-v6-axera-hw`
- **Local Build Directory**: `./realistic-vision-v6-axera-hw/`
- **License**: CreativeML Open RAIL-M (inherited from Stable Diffusion 1.5 and Realistic Vision)

## 2. Repository Structure
The following files will be organized in the `./realistic-vision-v6-axera-hw/` directory for upload:

```text
realistic-vision-v6-axera-hw/
├── README.md                  # The Model Card (detailed below)
├── LICENSE                    # CreativeML Open RAIL-M License
├── LICENSE_AXERA              # BSD 3-Clause License (for conversion tools)
├── Disclaimer.md              # Axera-specific safety and usage disclaimer
├── unet.axmodel               # Compiled U-Net (Fused with LCM-LoRA)
├── vae_decoder.axmodel        # Compiled VAE Decoder
├── vae_encoder.axmodel        # Compiled VAE Encoder
├── sd15_text_encoder_sim.axmodel # Compiled Text Encoder
├── sample_inference.py        # Minimal script to run inference on Axera hardware
└── assets/                    # Sample images and performance charts
```

## 3. Model Card (README.md) Content
The model card will include the following sections:

### A. Model Description
- **Base Model**: Realistic Vision V6.0 B1 (noVAE)
- **VAE**: sd-vae-ft-mse
- **Acceleration**: LCM-LoRA (Fused)
- **Target Hardware**: Axera AX650N / LLM8850 (NPU3)
- **Primary Use Case**: High-performance Stable Diffusion on **Raspberry Pi 5** (via Axera M.2 Accelerator Card)
- **Format**: Axera `.axmodel` (compiled via Pulsar2 v5.1)

### B. Performance Metrics
Detailed benchmarks from the test hardware:
- **Hardware**: Raspberry Pi 5 (8GB), Ubuntu 24.04 LTS, USB-3 SSD, Axera AX650N M.2 Accelerator.
- **Model Loading Time**: Time to initialize all 4 `.axmodel` files.
- **Inference Time**: Time per step and total time for a 512x512 image (4 steps).

### C. Sample Gallery
A diverse set of sample images generated on the target hardware:
- **Portraits**: Multiple styles (e.g., cinematic, oil painting, candid) representing diverse cultures and viewpoints.
- **Landscape**: High-detail natural scenery.
- **Object**: A complex 3D object (e.g., a vintage camera or a futuristic watch).
- *Note: All samples are generated to be non-objectifying and appropriate for all audiences.*

### D. Conversion Workflow Summary
- Exported from PyTorch/Diffusers to ONNX.
- Fused with LCM-LoRA weights for 4-8 step inference.
- **Graph Surgery**: Modified U-Net to expose the 1280-dim time embedding input (`/down_blocks.0/resnets.0/act_1/Mul_output_0`).
- **Quantization**: Compiled using Pulsar2 with u16/int8 mixed precision and representative calibration data.

### E. Usage Instructions
- Instructions on how to load the models using `axengine`.
- Reference to the `pi_axera_sd_generator` service for REST API access.
- Note on fixed resolution (512x512) and fixed steps (4).

### F. Credits and Citations
- **Realistic Vision**: [SG161222/Realistic_Vision_V6.0_B1_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE)
- **LCM-LoRA**: [latent-consistency/lcm-lora-sdv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)
- **Conversion Tools**: [AXERA-TECH/sd1.5-lcm.axera](https://github.com/AXERA-TECH/sd1.5-lcm.axera)
- **Optimization Work**: Compiled and optimized by [Your Name/Organization]

### G. Licensing and Restrictions
- Inclusion of the Open RAIL-M "Use Restrictions".
- Disclaimer regarding AI-generated content.

## 4. Action Checklist
- [ ] Copy `.axmodel` files from `models/rv_optimized/`.
- [ ] Create `README.md` based on the template above.
- [ ] Copy `LICENSE` and `Disclaimer.md` from the root project.
- [ ] Create a minimal `sample_inference.py` for users.
- [ ] Verify all files are present and correctly named.
- [ ] Initialize git repo and push to Hugging Face.
