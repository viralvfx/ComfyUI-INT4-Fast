import os
import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_detection
import comfy.lora
import comfy.lora_convert
import comfy.memory_management
import comfy.model_management
import logging

from .int4_quant import Int4Ops, INT4ModelPatcher


def _load_int4_unet_cached_patcher(unet_name, weight_dtype, model_type, on_the_fly_quantization, enable_convrot, lora_mode, pre_lora=None, disable_dynamic=False):
    return UNetLoaderINTW4A4().load_unet(
        unet_name,
        weight_dtype,
        model_type,
        on_the_fly_quantization,
        enable_convrot=enable_convrot,
        lora_mode=lora_mode,
        pre_lora=pre_lora,
        disable_dynamic=disable_dynamic,
    )[0]


class UNetLoaderINTW4A4:
    """
    Load INT4 convrot quantized diffusion models.
    
    Uses Int4Ops for direct INT4 loading and comfy-kitchen execution.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp16", "bf16", "fp32"], {"tooltip": "Compute dtype. Default follows the model dtype."}),
                "model_type": (["flux2", "z-image", "ideogram4", "chroma", "krea2", "wan", "ltx2", "qwen", "ernie", "anima", "hidream o1", "boogu"], {"tooltip": "Only used for on the fly quantization, to filter sensitive layers."}),
                "on_the_fly_quantization": ("BOOLEAN", {"default": False, "tooltip": "Quantize a higher precision model to INT4. If the selected model is already INT4 keep unchecked."}),
                "enable_convrot": ("BOOLEAN", {"default": True, "tooltip": "Enable ConvRot for better quantization."}),
                "lora_mode": (["None", "Stochastic", "Dynamic"], {"default": "None", "tooltip": "None bakes LoRA patches with normal rounding which is the default behavior. Stochastic bakes with stochastic rounding. Dynamic applies LoRA at inference time."}),
            },
            "optional": {
                "pre_lora": ("PRE_LORA",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"
    DESCRIPTION = "Load and Quantize INT4 models with fast comfy-kitchen GPU execution."

    def load_unet(self, unet_name, weight_dtype, model_type, on_the_fly_quantization, enable_convrot=True, lora_mode="None", pre_lora=None, disable_dynamic=False):
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)

        if isinstance(lora_mode, bool):
            lora_mode = "Dynamic" if lora_mode else "None"
        lora_mode = str(lora_mode)
        if lora_mode not in {"None", "Stochastic", "Dynamic"}:
            lora_mode = "None"
        
        if pre_lora is not None:
            loras_to_load = pre_lora if isinstance(pre_lora, list) else [pre_lora]
        else:
            loras_to_load = []
        
        model_options = {"custom_operations": Int4Ops}
        
        Int4Ops.excluded_names = []
        Int4Ops.dynamic_quantize = on_the_fly_quantization
        Int4Ops.enable_convrot = enable_convrot
        Int4Ops.convrot_groupsize = 256
        Int4Ops.quant_group_size = 64
        Int4Ops.linear_dtype = "int4"
        Int4Ops._is_prequantized = False
        Int4Ops.lora_mode = lora_mode
        Int4Ops.dynamic_lora = lora_mode == "Dynamic"
        Int4Ops.dynamic_load_device = None
        
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        Int4Ops.compute_dtype = dtype_map.get(str(weight_dtype), None)
        if comfy.memory_management.aimdo_enabled and (on_the_fly_quantization or len(loras_to_load) > 0):
            Int4Ops.dynamic_load_device = comfy.model_management.get_torch_device()
            logging.info(f"INT4 Fast: Aimdo dynamic loading active, using {Int4Ops.dynamic_load_device} as work device.")
        if hasattr(Int4Ops, "_logged_otf"):
            delattr(Int4Ops, "_logged_otf")
        
        if model_type == "flux2":
            Int4Ops.excluded_names = [
                'img_in', 'time_in', 'guidance_in', 'txt_in', 
                'double_stream_modulation_img', 'double_stream_modulation_txt', 
                'single_stream_modulation',
            ]
        elif model_type == "z-image":
            Int4Ops.excluded_names = [
                'cap_embedder', 't_embedder', 'x_embedder', 'cap_pad_token', 'context_refiner', 
                'final_layer', 'noise_refiner', 'adaLN',
                'x_pad_token', 'layers.0.',
            ]
        elif model_type == "chroma":
            Int4Ops.excluded_names = [
                'distilled_guidance_layer', 'final_layer', 'img_in', 'txt_in', 'nerf_image_embedder',
                 'nerf_blocks', 'nerf_final_layer_conv', '__x0__', 'nerf_final_layer_conv',
            ]
        elif model_type == "qwen":
            Int4Ops.excluded_names = [
                'time_text_embed', 'img_in', 'norm_out', 'proj_out', 'txt_in'
            ]
        elif model_type == "ernie":
            Int4Ops.excluded_names = [
                'time', 'x_embedder', 'text_proj', 'adaLN',
            ]
        elif model_type == "anima":
            Int4Ops.excluded_names = [
                'embed', 'llm', 'adaln',
            ]
        elif model_type == "krea2":
            Int4Ops.excluded_names = [
                'first', 'last', 'tmlp', 'tproj', 'txtfusion', 'txtmlp'
            ]
        elif model_type == "hidream o1":
            Int4Ops.excluded_names = [
                'embed', 'language_model.layers.35.mlp',
            ]
        elif model_type == "boogu":
            Int4Ops.excluded_names = [
                'embed', 'refine', 'norm_out',
            ]
        elif model_type == "ideogram4":
            Int4Ops.excluded_names = [
                'embed_image_indicator', 't_embedding', 'proj',
            ]
        elif model_type == "wan":
            Int4Ops.excluded_names = [
                'patch_embedding', 'text_embedding', 'time_embedding', 'time_projection', 'head',
                'img_emb', 'face_adapter', 'face_encoder', 'motion_encoder', 'pose_patch_embedding',
            ]
        elif model_type == "ltx2":
            Int4Ops.excluded_names = [
                'adaln', 'embedding', 'patchify', 'to_gate_logits', 'proj_out', 'model.audio', 'model.video', 'model.av', 'model.patch', 'model.proj', 'shift',
            ]

        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
        
        Int4Ops.lora_patches = {}
        if len(loras_to_load) > 0:
            grouped_patches = {}
            for lora in loras_to_load:
                lora_name = lora.get("lora_name", "None")
                lora_strength = lora.get("lora_strength", 1.0)
                
                if lora_name == "None":
                    continue
                    
                lora_path = folder_paths.get_full_path("loras", lora_name)
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                lora_data = comfy.lora_convert.convert_lora(lora_data)
                
                unet_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
                m_config = comfy.model_detection.model_config_from_unet(sd, unet_prefix, metadata=metadata)
                
                if m_config is None and unet_prefix != "":
                    m_config = comfy.model_detection.model_config_from_unet(sd, "", metadata=metadata)
                    if m_config is not None:
                        unet_prefix = ""

                if m_config is not None:
                    m_config.custom_operations = Int4Ops
                    Int4Ops.skeleton_meta_init = True
                    try:
                        skeleton_model = m_config.get_model(sd, unet_prefix)
                    finally:
                        Int4Ops.skeleton_meta_init = False
                    key_map = comfy.lora.model_lora_keys_unet(skeleton_model, {})

                    patch_dict = comfy.lora.load_lora(lora_data, key_map)
                    
                    def normalize_key(key):
                        if not isinstance(key, str):
                            return key
                        for p in ["diffusion_model.", "model.diffusion_model.", "model.", "transformer."]:
                            if key.startswith(p):
                                return key[len(p):]
                        return key
                    
                    for k, v in patch_dict.items():
                        target_key = k
                        offset = None
                        function = None
                        if isinstance(k, tuple):
                            target_key = k[0]
                            if len(k) > 1: offset = k[1]
                            if len(k) > 2: function = k[2]
                        
                        nk = normalize_key(target_key)
                        if nk not in grouped_patches:
                            grouped_patches[nk] = []
                        grouped_patches[nk].append((v, offset, function, lora_strength))
                else:
                    logging.warning(f"INT4 Fast: Could not detect model type for LoRA mapping.")
                
                del lora_data
            
            if grouped_patches:
                Int4Ops.lora_patches = grouped_patches
                logging.info(f"INT4 Fast: Prepared {len(grouped_patches)} layer patches for baking.")
            
        try:
            Int4Ops.applied_lora_patches = set()
            model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata, disable_dynamic=disable_dynamic)
            
            if Int4Ops.lora_patches:
                unmatched = set(Int4Ops.lora_patches.keys()) - Int4Ops.applied_lora_patches
                if unmatched:
                    print(f"INT4 Fast: {len(unmatched)} LoRA keys were NOT matched:")
                    for k in sorted(unmatched):
                        print(f"  unmatched: {k}")
        finally:
            dynamic_load_device = Int4Ops.dynamic_load_device
            Int4Ops.lora_patches = {}
            Int4Ops.dynamic_load_device = None
            if dynamic_load_device is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(Int4Ops, 'applied_lora_patches'):
                delattr(Int4Ops, 'applied_lora_patches')
        
        model = INT4ModelPatcher.clone(model)
        model.cached_patcher_init = (
            _load_int4_unet_cached_patcher,
            (unet_name, weight_dtype, model_type, on_the_fly_quantization, enable_convrot, lora_mode, pre_lora),
        )
        model._safetensors_metadata = metadata
        try:
            if model.model is not None:
                model.model._int4_source_metadata = metadata
        except Exception:
            pass

        return (model,)
