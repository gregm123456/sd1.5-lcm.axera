# SD1.5-LCM.Axera

This project demonstrates how to deploy the StableDiffusion 1.5 LCM (Latent Consistency Model) project on Axera hardware.

Supported chip:

- AX650N

Original models referenced:

- Latent Consistency Model (LCM) LoRA: SDv1-5: https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
- Dreamshaper 7: https://huggingface.co/Lykon/dreamshaper-7

Clone the repository:

```sh
git clone https://github.com/AXERA-TECH/sd1.5-lcm.axera
cd sd1.5-lcm.axera
```

The `model_convert` folder contains tools and scripts for compiling and converting models — see the model conversion documentation at `model_convert/README.md`. The `python` folder contains inference scripts.

## ONNX and AXMODEL Inference

After you finish converting and compiling model files, go to the `python` folder to run text-to-image or image-to-image tasks.

```sh
cd sd1.5-lcm.axera/python
```

A typical `models/` layout looks like this:

```sh
ai@ai-bj ~/yongqiang/sd1.5-lcm.axera/python $ tree -L 2 models/
models/
├── 7ffcf62c-d292-11ef-bb2a-9d527016cd35
├── text_encoder
│   ├── config.json
│   ├── model.fp16.safetensors
│   ├── model.safetensors
│   ├── sd15_text_encoder_sim.axmodel
│   └── sd15_text_encoder_sim.onnx
├── time_input_img2img.npy
├── time_input_txt2img.npy
├── tokenizer
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── unet.axmodel
├── vae_decoder.axmodel
└── vae_encoder.axmodel

2 directories, 15 files
```

With this layout you can run ONNX or AXMODEL inference for text-to-image and image-to-image tasks using the provided scripts:

```sh
python3 run_txt2img_axe[onnx]_infer.py --prompt "your prompt"
python3 run_img2img_axe[onnx]_infer.py --prompt "your prompt"
```

### Running on Axera

On an Axera development board, inference uses the precompiled `axmodel` files. The main scripts are `run_txt2img_axe_infer.py` (text-to-image) and `run_img2img_axe_infer.py` (image-to-image).

#### Text-to-Image Example

Input prompt example:

```
"masterpiece, best quality, serene mountain lake at sunrise, ultra-detailed landscape, soft pastel colors, fog on the water, pine trees, reflection, 8k"
```

Example output log:

```sh
ai@ai-bj ~/yongqiang/sd1.5-lcm.axera/python $ python3 run_txt2img_axe_infer.py --prompt "masterpiece, best quality, serene mountain lake at sunrise, ultra-detailed landscape, soft pastel colors, fog on the water, pine trees, reflection, 8k"
[INFO] Available providers:  ['AXCLRTExecutionProvider']
prompt: masterpiece, best quality, serene mountain lake at sunrise, ultra-detailed landscape, soft pastel colors, fog on the water, pine trees, reflection, 8k
text_tokenizer: ./models/tokenizer
text_encoder: ./models/text_encoder
unet_model: ./models/unet.axmodel
vae_decoder_model: ./models/vae_decoder.axmodel
time_input: ./models/time_input_txt2img.npy
save_dir: ./txt2img_output_axe.png
[INFO] Using provider: AXCLRTExecutionProvider
[INFO] SOC Name: AX650N
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Compiler version: 3.4 9215b7e5
text encoder axmodel take 9.8ms
get_embeds take 11.5ms
[INFO] Using provider: AXCLRTExecutionProvider
[INFO] SOC Name: AX650N
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Compiler version: 3.3 972f38ca
[INFO] Using provider: AXCLRTExecutionProvider
[INFO] SOC Name: AX650N
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Compiler version: 3.3 972f38ca
load models take 15280.1ms
unet once take 436.0ms
unet once take 437.8ms
unet once take 437.5ms
unet once take 437.8ms
unet loop take 1753.4ms
vae inference take 930.4ms
save image take 123.3ms
```

Output image:

![](assets/txt2img_output_axe.png)

#### Image-to-Image Example

Input image and prompt:

Provide an initial image and prompt. Example initial image:

![](assets/img2img-init.png)

Run:

```sh
python3 run_img2img_axe_infer.py --init_image models/img2img-init.png --prompt "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"
```

Output image:

![](assets/lcm_lora_sdv1-5_imgGrid_output.png)

The right-hand image is the image-to-image result.

## Related Projects

NPU toolchain: Pulsar2 documentation — https://pulsar2-docs.readthedocs.io/zh-cn/latest/

## Discussion

Open an issue on GitHub or join the QQ group: 139953715

## Disclaimer

- This project is intended only as a guide for deploying the Latent Consistency Model (LCM) LoRA (SDv1-5) open-source models on AX650N.
- The models have inherent limitations and may produce incorrect, harmful, offensive, or otherwise undesirable outputs; such outputs are unrelated to AX650N hardware or the repository owners.
- See the full disclaimer: ./Disclaimer.md
