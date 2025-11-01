# Model Conversion

Complete ONNX model export and axmodel compilation on PC. Note that to obtain the toolchain `docker image`, you need to follow the official release process.

## Install Dependencies

```
git clone https://github.com/AXERA-TECH/sd1.5-lcm.axera.git
cd model_convert
pip install -r requirements.txt
```

## Export Models (Huggingface -> ONNX)

Download the corresponding Repo from Huggingface.

```sh
$ huggingface-cli download --resume-download latent-consistency/lcm-lora-sdv1-5 --local-dir latent-consistency/lcm-lora-sdv1-5

$ huggingface-cli download --resume-download Lykon/dreamshaper-7 --local-dir Lykon/dreamshaper-7
```

Run the script `sd15_export_onnx.py` to export the `text_encoder`, `unet`, and `vae` ONNX models.

```sh
python3 sd15_export_onnx.py --input_path ./hugging_face/models/dreamshaper-7/ --input_lora_path ./hugging_face/models/lcm-lora-sdv1-5/ --output_path onnx-models
```

The default exported `vae_encoder` model input image size is `512x512`. If you need other sizes, you can use `--isize 256x256` in the command line to modify the exported size.

Exporting takes some time, please be patient.

The exported file directory is as follows:

```sh
✗ tree -L 1 onnx-models
onnx-models
├── a9a1a634-4cf5-11f0-b3ee-f5b7bf5aa809
├── sd15_text_encoder_sim.onnx
├── time_input_img2img.npy # Note to use different time inputs for different tasks
├── time_input_txt2img.npy
├── unet.onnx
├── vae_decoder.onnx
└── vae_encoder.onnx

0 directories, 7 files
```

Note: If using the **latest version** of the toolchain, when compiling models, if you encounter the following error:

```sh
Traceback (most recent call last):
  File "/home/baiyongqiang/local_space/npu-codebase/frontend/graph_ir.py", line 999, in shapefn
    outputs_spec = self.impl.shapefn(inputs_tinfo)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/baiyongqiang/local_space/npu-codebase/opset/oprdef.py", line 132, in shapefn
    outputs_shapes = self._shapefn(self._attrs, inputs_spec)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/baiyongqiang/local_space/npu-codebase/frontend/operators/onnx/onnx_ops.py", line 1030, in <lambda>
    .setShapeInference(lambda attrs, inputs: onnx_shapefn_or_pyrun(attrs, inputs, True, "Reshape"))
                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/baiyongqiang/local_space/npu-codebase/frontend/operators/onnx/utils.py", line 159, in onnx_shapefn_or_pyrun
    model, inputs_data = make_one_node_model(attrs, inputs, outputs, op_type)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/baiyongqiang/local_space/npu-codebase/frontend/operators/onnx/utils.py", line 79, in make_one_node_model
    assert v.is_const and v.data is not None
AssertionError
```

You can manually run the `onnxslim` command to optimize the exported ONNX model and resolve this error. Example command:

```bash
pip3 install onnxslim
onnxslim vae_encoder.onnx vae_encoder_slim.onnx
```

## Generate Quantization Dataset

Run the script `sd15_lora_prepare_data.py` to prepare the `Calibration` dataset required by `Pulsar2` compilation.

```sh
python3 sd15_lora_prepare_data.py --export_onnx_dir onnx-models[onnx export directory]
```

After the code execution, enter the `datasets` directory, and you can observe the directory structure as follows:

```sh
datasets git:(yongqiang/dev) ✗ tree -L 1 calib_data_unet
calib_data_unet
├── data_0.npy
......
├── data_9.npy
└── data.tar
datasets git:(yongqiang/dev) ✗ tree -L 1 calib_data_vae
calib_data_vae
├── data_0_0.npy
......
├── data_9_3.npy
└── data.tar
```

## Model Conversion

In the `Axera` toolchain `docker`, execute the following commands respectively for model compilation.

(1) Compile the `sd15_text_encoder_sim.onnx` model

```sh
pulsar2 build --input onnx-models/sd15_text_encoder_sim.onnx  --output_dir axmodels  --output_name sd15_text_encoder_sim.axmodel --config configs/text_encoder_u16.json --quant.precision_analysis 1 --quant.precision_analysis_method EndToEnd
```


(2) Compile the `vae_encoder.onnx` model

```sh
pulsar2 build --input onnx-models/vae_encoder.onnx  --output_dir axmodels  --output_name vae_encoder.axmodel --config configs/vae_encoder_u16.json --quant.precision_analysis 1 --quant.precision_analysis_method EndToEnd
```


(3) Compile the `vae_decoder.onnx` model

```sh
pulsar2 build --input onnx-models/vae_decoder.onnx  --output_dir axmodels  --output_name vae_decoder.axmodel --config configs/vae_u16.json --quant.precision_analysis 1 --quant.precision_analysis_method EndToEnd
```

(4) Compile the `unet.onnx` model

```sh
pulsar2 build  --input onnx-models/unet.onnx  --output_dir axmodels  --output_name unet.axmodel --config configs/unet_u16.json --quant.precision_analysis 1 --quant.precision_analysis_method EndToEnd
```

