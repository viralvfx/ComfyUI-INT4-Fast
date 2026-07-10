import folder_paths
import comfy.sd
import comfy.model_patcher
import comfy.model_management
import json
import os
import logging
import torch
from comfy.cli_args import args


def _is_dynamic_lora_enabled():
    try:
        from .int4_quant import Int4Ops
        return bool(getattr(Int4Ops, "dynamic_lora", False))
    except Exception:
        return False


def _resolve_source_metadata(model):
    seen = set()

    def _walk(m):
        if m is None or id(m) in seen:
            return None
        seen.add(id(m))
        meta = getattr(m, "_safetensors_metadata", None)
        if isinstance(meta, dict) and meta:
            return meta
        inner = getattr(m, "model", None)
        if inner is not None:
            inner_meta = getattr(inner, "_int4_source_metadata", None)
            if not inner_meta:
                inner_meta = getattr(inner, "_int8_source_metadata", None)
            if isinstance(inner_meta, dict) and inner_meta:
                return inner_meta
        parent = getattr(m, "parent", None)
        return _walk(parent)

    return _walk(model)


class INT4ModelSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "filename_prefix": ("STRING", {"default": "int4_models/INT4_Model"}),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "loaders"

    def save(self, model, filename_prefix, prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        metadata = {}
        src_meta = _resolve_source_metadata(model)
        if isinstance(src_meta, dict):
            metadata.update(src_meta)
        if not src_meta:
            logging.warning(
                "INT4 Save: source safetensors metadata could not be located on the patcher chain. "
                "The output checkpoint will be saved without int4_quantized/int4_model_type/config metadata."
            )

        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        extra_keys = {}
        patched_modules = []
        patched_module_ids = set()

        def mark_module_for_direct_save(module):
            module_id = id(module)
            if module_id in patched_module_ids:
                return
            had_flag = hasattr(module, "comfy_patched_weights")
            old_flag = getattr(module, "comfy_patched_weights", False)
            patched_modules.append((module, had_flag, old_flag))
            patched_module_ids.add(module_id)
            module.comfy_patched_weights = True

        def module_has_quantized_param(module):
            from comfy.quant_ops import QuantizedTensor
            for attr in ("weight", "bias"):
                tensor = getattr(module, attr, None)
                if isinstance(tensor, QuantizedTensor):
                    return True
                if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.int8:
                    return True
            return False

        def iter_model_modules(model_patcher):
            if hasattr(model_patcher, "model") and hasattr(model_patcher.model, "named_modules"):
                yield from model_patcher.model.named_modules()

        def materialize_int4_lora_patches(model_patcher):
            if _is_dynamic_lora_enabled() or not hasattr(model_patcher, "patch_weight_to_device"):
                return

            patches = getattr(model_patcher, "patches", None)
            if not patches:
                return

            load_device = getattr(model_patcher, "load_device", None)
            materialized = 0
            for name, module in iter_model_modules(model_patcher):
                if not getattr(module, "_is_quantized", False):
                    continue

                weight_key = name + ".weight" if name else "weight"
                if weight_key not in patches:
                    continue

                try:
                    current_weight = getattr(module, "weight", None)
                    device_to = load_device if load_device is not None else getattr(current_weight, "device", None)
                    model_patcher.patch_weight_to_device(weight_key, device_to=device_to)
                    if hasattr(module, "weight_lowvram_function"):
                        module.weight_lowvram_function = None
                    materialized += 1
                except Exception as e:
                    logging.warning(
                        f"INT4 Save: failed to materialize LoRA patch for {weight_key}: {e}."
                    )

            if materialized > 0:
                logging.info(f"INT4 Save: materialized {materialized} INT4 LoRA patched weight(s) before saving.")

        finalize_fn = getattr(model, "finalize_pending_int8", None)
        if finalize_fn is not None:
            finalize_fn()

        try:
            comfy.model_management.load_models_gpu([model], force_full_load=True)
        except Exception as e:
            logging.warning(
                f"INT4 Save: full-load pre-pass failed ({e}); falling back to default load_models_gpu."
            )
            try:
                comfy.model_management.load_models_gpu([model])
            except Exception as e2:
                logging.warning(
                    f"INT4 Save: load_models_gpu fallback also failed ({e2}); continuing best-effort."
                )

        if finalize_fn is not None:
            finalize_fn()

        materialize_int4_lora_patches(model)

        if hasattr(model, "model"):
            for name, module in iter_model_modules(model):
                if module_has_quantized_param(module):
                    mark_module_for_direct_save(module)

                if getattr(module, "_is_quantized", False):
                    quant_conf = {
                        "format": "convrot_w4a4",
                        "convrot_groupsize": int(getattr(module, "_convrot_groupsize", 256)),
                        "quant_group_size": int(getattr(module, "_quant_group_size", 64)),
                        "linear_dtype": getattr(module, "_linear_dtype", "int4"),
                    }
                    
                    prefix = "model." + name + "." if name else "model."
                    
                    extra_keys[prefix + "comfy_quant"] = torch.tensor(
                        list(json.dumps(quant_conf).encode('utf-8')), dtype=torch.uint8
                    )
                    mark_module_for_direct_save(module)

        original_lazy_new = comfy.model_patcher.LazyCastingParam.__new__
        original_lazy_piece_new = comfy.model_patcher.LazyCastingParamPiece.__new__

        def lazy_casting_param_new(cls, model, key, tensor):
            requires_grad = tensor.is_floating_point() or tensor.is_complex()
            return torch.nn.Parameter.__new__(cls, tensor, requires_grad=requires_grad)

        def lazy_casting_param_piece_new(cls, caster, state_dict_key, tensor):
            requires_grad = tensor.is_floating_point() or tensor.is_complex()
            return torch.nn.Parameter.__new__(cls, tensor, requires_grad=requires_grad)

        original_state_dict_for_saving = model.state_dict_for_saving

        def custom_state_dict_for_saving(*args, **kwargs):
            from comfy.quant_ops import QuantizedTensor
            sd = original_state_dict_for_saving(*args, **kwargs)
            new_sd = {}
            for k, v in sd.items():
                if isinstance(v, QuantizedTensor) or (hasattr(v, "_qdata") and hasattr(v, "_params")):
                    new_sd[k] = v._qdata.cpu()
                    scale_key = k.rsplit('.', 1)[0] + ".weight_scale"
                    scale_val = v._params.scale
                    if isinstance(scale_val, torch.Tensor):
                        scale_val = scale_val.reshape(-1).cpu()
                    new_sd[scale_key] = scale_val
                elif hasattr(v, "weight") and (isinstance(v.weight, QuantizedTensor) or (hasattr(v.weight, "_qdata") and hasattr(v.weight, "_params"))):
                    new_sd[k] = v.weight._qdata.cpu()
                    scale_key = k.rsplit('.', 1)[0] + ".weight_scale"
                    scale_val = v.weight._params.scale
                    if isinstance(scale_val, torch.Tensor):
                        scale_val = scale_val.reshape(-1).cpu()
                    new_sd[scale_key] = scale_val
                else:
                    new_sd[k] = v
            return new_sd

        had_save_flag = hasattr(model, "_int4_save_materialized_lora")
        old_save_flag = getattr(model, "_int4_save_materialized_lora", False)

        try:
            model._int4_save_materialized_lora = True
            comfy.model_patcher.LazyCastingParam.__new__ = staticmethod(lazy_casting_param_new)
            comfy.model_patcher.LazyCastingParamPiece.__new__ = staticmethod(lazy_casting_param_piece_new)
            model.state_dict_for_saving = custom_state_dict_for_saving
            comfy.sd.save_checkpoint(output_checkpoint, model, metadata=metadata, extra_keys=extra_keys)
        finally:
            model.state_dict_for_saving = original_state_dict_for_saving
            comfy.model_patcher.LazyCastingParam.__new__ = original_lazy_new
            comfy.model_patcher.LazyCastingParamPiece.__new__ = original_lazy_piece_new
            if had_save_flag:
                model._int4_save_materialized_lora = old_save_flag
            else:
                try:
                    delattr(model, "_int4_save_materialized_lora")
                except AttributeError:
                    pass
            for module, had_flag, old_flag in patched_modules:
                if had_flag:
                    module.comfy_patched_weights = old_flag
                else:
                    try:
                        delattr(module, "comfy_patched_weights")
                    except AttributeError:
                        pass

        return {}
