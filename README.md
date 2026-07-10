# ComfyUI-INT4-Fast

A high-performance ComfyUI custom node package designed to load, run, and serialize diffusion models in **INT4 format** (`convrot_w4a4` layout). It leverages Tensor Cores for ultra-fast, memory-efficient inference.

This project is built on top of the original work by **BobJohnson24** (author of [ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast)), adapting the core dynamic quantization, Hadamard rotation, and model patching architectures to support native INT4 workflows.

---

## Features

- **Fast INT4 Inference (`convrot_w4a4`)**: Load and run models quantized to 4-bit weights with 4-bit activation layouts, leveraging GPU Tensor Cores.
- **Mixed-Precision Checkpoint Support**: Automatically parses `comfy_quant` metadata on a per-layer basis to route standard INT4 layers to Tensor Cores, and sensitive layers (like `first` and `last` patch projections stored in INT8 format) to optimized Triton execution paths.
- **On-the-Fly Quantization**: Instantly quantizes standard float (BF16/FP16/FP32) checkpoints to INT4 upon loading.
- **Dynamic LoRA Patching**: Automatically handles LoRA injection, dynamically applying Hadamard rotation to the LoRA `down` projection weights so they remain coherent with the rotated weight basis.
- **Checkpoint Serialization**: Saves quantized model checkpoints in the standard ComfyUI quantized model format.

---

## Nodes Included

1. **Load Diffusion Model INT4 (W4A4)** (`OTUNetLoaderW4A4`):
   - Loads pre-quantized or float diffusion checkpoints.
   - Allows enabling/disabling Hadamard rotation (`enable_convrot`), configuring the rotation group size (`convrot_groupsize`), and specifying linear execution precision.
2. **Save Int4 Model** (`INT4ModelSave`):
   - Compiles and serializes model checkpoints to the standard INT4 format.

---

## Verified Models

This package has been verified and tested using the following pre-quantized INT4/INT8 mixed-precision model:
- **Krea2 Turbo INT4**: [krea2_turbo_convrot_int4_fast.safetensors](https://huggingface.co/comfyanonymous/int4_tests/blob/main/split_files/diffusion_models/krea2_turbo_convrot_int4_fast.safetensors)

> [!NOTE]
> The **first generation run** will take some additional time during model initialization to compile and set up the custom operators. Subsequent generations will run significantly faster.

---

## Installation & Setup

To install this custom node, navigate to your ComfyUI `custom_nodes` directory and clone the repository:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/viralvfx/ComfyUI-INT4-Fast.git
```

Make sure you have:
1. **ComfyUI updated to the latest version**.
2. **`comfy-kitchen` installed**, as it provides the core `QuantizedTensor` execution layouts.

---

## Credits & Attributions

Special thanks to **BobJohnson24** for the excellent codebase and architectural design of [ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast), which served as the structural foundation for this project.
