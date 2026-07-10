import torch
from torch import Tensor, nn
import torch.nn.functional as F
import logging
import json
import comfy.model_patcher
import comfy.memory_management
import comfy.model_management
import comfy.lora
import comfy.utils
from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor, TensorCoreConvRotW4A4Layout

from .int8_quant import (
    int8_forward_dynamic,
    int8_forward_dynamic_per_row,
    dequantize as dequantize_int8,
    Int8TensorwiseOps
)

_use_new_prefetch = False
import inspect
try:
    _prefetch_sig = inspect.signature(comfy.lora.prefetch_prepared_value)
    _use_new_prefetch = len(_prefetch_sig.parameters) == 5
except Exception:
    pass

fp8_types = []
for tname in ("float8_e4m3fn", "float8_e5m2"):
    if hasattr(torch, tname):
        fp8_types.append(getattr(torch, tname))

class Int4Ops(manual_cast):
    excluded_names = []
    dynamic_quantize = False
    enable_convrot = True
    convrot_groupsize = 256
    quant_group_size = 64
    linear_dtype = "int4"
    lora_mode = "None"
    dynamic_lora = False
    lora_patches = {}
    dynamic_load_device = None
    skeleton_meta_init = False
    _is_prequantized = False

    class Linear(manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            if getattr(Int4Ops, "skeleton_meta_init", False):
                nn.Module.__init__(self)
                self.in_features = in_features
                self.out_features = out_features
                tensor_kwargs = {"device": "meta"}
                if dtype is not None:
                    tensor_kwargs["dtype"] = dtype
                self.weight = nn.Parameter(torch.empty((out_features, in_features), **tensor_kwargs), requires_grad=False)
                self.bias = nn.Parameter(torch.empty((out_features,), **tensor_kwargs), requires_grad=False) if bias else None
                self.weight_comfy_model_dtype = dtype
                self.bias_comfy_model_dtype = dtype
            elif comfy.model_management.WINDOWS and comfy.memory_management.aimdo_enabled:
                nn.Module.__init__(self)
                self.in_features = in_features
                self.out_features = out_features
                self.weight = None
                self.bias = None
                self.comfy_need_lazy_init_bias = bias
                self.weight_comfy_model_dtype = dtype
                self.bias_comfy_model_dtype = dtype
            else:
                super().__init__(in_features, out_features, bias, device, dtype)
            
            self._is_quantized = False
            self._quant_format = "convrot_w4a4"
            self._convrot_groupsize = 256
            self._quant_group_size = 64
            self._linear_dtype = "int4"
            self._is_per_row = False
            self._use_convrot = False
            self.comfy_cast_weights = False
            self.lora_patches = []

        def reset_parameters(self):
            return None

        @staticmethod
        def _normalize_lora_key(key):
            if not isinstance(key, str):
                return key
            for p in ["diffusion_model.", "model.diffusion_model.", "model.", "transformer."]:
                if key.startswith(p):
                    return key[len(p):]
            return key

        @staticmethod
        def _is_bias_key(key):
            return isinstance(key, str) and key.endswith(".bias")

        def finalize_pending_int8(self):
            pending = getattr(self, "_pending_int4_finalize", None)
            if pending is None:
                return False

            weight_key = pending["weight_key"]
            device = pending.get("device")
            if device is None:
                device = torch.device("cuda") if torch.cuda.is_available() else self.weight.device

            weight_tensor = self.weight
            if isinstance(weight_tensor, QuantizedTensor) or (hasattr(weight_tensor, "_qdata") and hasattr(weight_tensor, "_params")):
                weight_tensor = weight_tensor.dequantize()

            if hasattr(weight_tensor, "to"):
                weight_tensor = weight_tensor.to(device=device)
            else:
                weight_tensor = weight_tensor.detach()

            if pending.get("cast_fp8") or weight_tensor.dtype in fp8_types:
                weight_tensor = weight_tensor.to(dtype=torch.bfloat16)

            if pending.get("lora_patches"):
                temp_dtype = comfy.model_management.lora_compute_dtype(device)
                w_temp = weight_tensor.to(device=device, dtype=temp_dtype)
                formatted = []
                for patch in pending["lora_patches"]:
                    if len(patch) == 4:
                        v, offset, function, strength = patch
                    else:
                        v, offset, function = patch
                        strength = 1.0
                    formatted.append((strength, v, 1.0, offset, function))
                weight_tensor = comfy.lora.calculate_weight(formatted, w_temp, weight_key).cpu()

            if pending["quantize"]:
                w_gpu = weight_tensor.to(device).float()
                stochastic_rounding = 1 if getattr(Int4Ops, "lora_mode", "None") == "Stochastic" else 0
                
                qdata, params = TensorCoreConvRotW4A4Layout.quantize(
                    w_gpu,
                    convrot_groupsize=pending.get("convrot_groupsize", 256),
                    quant_group_size=pending.get("quant_group_size", 64),
                    stochastic_rounding=stochastic_rounding,
                    linear_dtype=pending.get("linear_dtype", "int4"),
                )
                
                params_cpu = TensorCoreConvRotW4A4Layout.Params(
                    scale=params.scale.cpu() if isinstance(params.scale, torch.Tensor) else params.scale,
                    orig_dtype=params.orig_dtype,
                    orig_shape=params.orig_shape,
                    convrot_groupsize=params.convrot_groupsize,
                    quant_group_size=params.quant_group_size,
                    linear_dtype=params.linear_dtype,
                )
                
                self.weight = nn.Parameter(QuantizedTensor(qdata.cpu(), "TensorCoreConvRotW4A4Layout", params_cpu), requires_grad=False)
                self._is_quantized = True
                self._quant_format = "convrot_w4a4"
                self._convrot_groupsize = params.convrot_groupsize
                self._quant_group_size = params.quant_group_size
                self._linear_dtype = params.linear_dtype
                del w_gpu, qdata
            else:
                self.weight = nn.Parameter(weight_tensor.cpu(), requires_grad=False)

            self.weight_comfy_model_dtype = self.weight.dtype
            if self.bias is not None:
                self.bias_comfy_model_dtype = self.bias.dtype

            delattr(self, "_pending_int4_finalize")
            return True

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            weight_key = prefix + "weight"
            bias_key = prefix + "bias"

            def normalize_key(key):
                return self._normalize_lora_key(key)

            def pop_metadata(sd, p, k):
                v = sd.pop(p + k, None)
                if v is not None: return v
                v = sd.pop("model." + p + k, None)
                if v is not None: return v
                if p.startswith("model."):
                    v = sd.pop(p[6:] + k, None)
                    if v is not None: return v
                if p.startswith("diffusion_model."):
                    v = sd.pop("diffusion_model." + p + k, None)
                    if v is not None: return v
                return None

            weight_scale = pop_metadata(state_dict, prefix, "weight_scale")
            comfy_quant_tensor = pop_metadata(state_dict, prefix, "comfy_quant")

            weight_tensor = state_dict.pop(weight_key, None)
            bias_tensor = state_dict.pop(bias_key, None)

            is_fp8 = weight_tensor is not None and weight_tensor.dtype in fp8_types

            quant_conf_parsed = None
            quant_format = "convrot_w4a4"
            if comfy_quant_tensor is not None:
                try:
                    quant_conf_parsed = json.loads(bytes(comfy_quant_tensor.tolist()).decode('utf-8'))
                    quant_format = quant_conf_parsed.get("format", "convrot_w4a4")
                    if "convrot_groupsize" in quant_conf_parsed:
                        self._convrot_groupsize = int(quant_conf_parsed["convrot_groupsize"])
                    if "quant_group_size" in quant_conf_parsed:
                        self._quant_group_size = int(quant_conf_parsed["quant_group_size"])
                    if "linear_dtype" in quant_conf_parsed:
                        self._linear_dtype = quant_conf_parsed["linear_dtype"]
                    if quant_conf_parsed.get("convrot", False):
                        self._use_convrot = True
                except Exception:
                    pass

            self._quant_format = quant_format

            pending_weight_lora_patches = None
            if weight_tensor is not None and not isinstance(weight_tensor, QuantizedTensor) and weight_tensor.dtype != torch.int8:
                pending_weight_lora_patches = Int4Ops.lora_patches.get(normalize_key(weight_key))

            defer_weight_lora = (
                getattr(Int4Ops, "dynamic_load_device", None) is not None
                and pending_weight_lora_patches
            )

            if weight_tensor is not None and not defer_weight_lora and pending_weight_lora_patches:
                device = getattr(Int4Ops, "dynamic_load_device", None)
                if device is None:
                    device = weight_tensor.device

                if hasattr(weight_tensor, "to"):
                    weight_tensor_resolved = weight_tensor.to(device=device)
                else:
                    weight_tensor_resolved = weight_tensor

                if isinstance(weight_tensor_resolved, QuantizedTensor) or (hasattr(weight_tensor_resolved, "_qdata") and hasattr(weight_tensor_resolved, "_params")):
                    weight_tensor_resolved = weight_tensor_resolved.dequantize()

                if is_fp8 or weight_tensor_resolved.dtype in fp8_types:
                    weight_tensor_resolved = weight_tensor_resolved.to(dtype=torch.bfloat16)

                temp_dtype = comfy.model_management.lora_compute_dtype(device)
                w_temp = weight_tensor_resolved.to(device=device, dtype=temp_dtype)
                
                formatted = []
                for patch in pending_weight_lora_patches:
                    if len(patch) == 4:
                        v, offset, function, strength = patch
                    else:
                        v, offset, function = patch
                        strength = 1.0
                    formatted.append((strength, v, 1.0, offset, function))
                weight_tensor = comfy.lora.calculate_weight(formatted, w_temp, weight_key).cpu()
                if not hasattr(Int4Ops, 'applied_lora_patches'):
                    Int4Ops.applied_lora_patches = set()
                Int4Ops.applied_lora_patches.add(normalize_key(weight_key))
                is_fp8 = False

            if bias_tensor is not None:
                bias_patches = Int4Ops.lora_patches.get(normalize_key(bias_key))
                if bias_patches:
                    device = getattr(Int4Ops, "dynamic_load_device", None)
                    if device is None:
                        device = bias_tensor.device
                    temp_dtype = comfy.model_management.lora_compute_dtype(device)
                    b_temp = bias_tensor.to(device=device, dtype=temp_dtype)
                    formatted = []
                    for patch in bias_patches:
                        if len(patch) == 4:
                            v, offset, function, strength = patch
                        else:
                            v, offset, function = patch
                            strength = 1.0
                        formatted.append((strength, v, 1.0, offset, function))
                    bias_tensor = comfy.lora.calculate_weight(formatted, b_temp, bias_key).cpu()
                    if not hasattr(Int4Ops, 'applied_lora_patches'):
                        Int4Ops.applied_lora_patches = set()
                    Int4Ops.applied_lora_patches.add(normalize_key(bias_key))

            def source_tensor(tensor):
                if tensor is not None and getattr(Int4Ops, "dynamic_load_device", None) is not None:
                    return tensor.cpu()
                return tensor

            if weight_tensor is not None:
                if (weight_tensor.dtype == torch.int8 or isinstance(weight_tensor, QuantizedTensor)) and weight_scale is not None:
                    self._is_quantized = True
                    Int4Ops._is_prequantized = True
                    
                    if quant_format == "int8_tensorwise":
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        per_row_hint = None
                        if isinstance(quant_conf_parsed, dict) and "per_row" in quant_conf_parsed:
                            per_row_hint = bool(quant_conf_parsed["per_row"])

                        if isinstance(weight_scale, torch.Tensor):
                            if weight_scale.numel() == 1:
                                self.register_buffer('weight_scale', weight_scale.float().reshape(1))
                                self._is_per_row = False
                            elif weight_scale.dim() == 2 and weight_scale.shape[1] == 1:
                                self.register_buffer('weight_scale', weight_scale.float())
                                self._is_per_row = True if per_row_hint is None else per_row_hint
                            else:
                                self.register_buffer('weight_scale', weight_scale.float())
                                self._is_per_row = False if per_row_hint is None else per_row_hint
                        else:
                            self.weight_scale = nn.Parameter(
                                torch.tensor(float(weight_scale), dtype=torch.float32), 
                                requires_grad=False
                            )
                            self._is_per_row = False if per_row_hint is None else per_row_hint
                    else:
                        orig_dtype = getattr(self, "weight_comfy_model_dtype", torch.bfloat16)
                        if orig_dtype is None:
                            orig_dtype = torch.bfloat16
                        orig_shape = (self.out_features, self.in_features)
                        
                        if isinstance(weight_scale, torch.Tensor):
                            weight_scale = weight_scale.reshape(-1)
                        
                        params = TensorCoreConvRotW4A4Layout.Params(
                            scale=weight_scale,
                            orig_dtype=orig_dtype,
                            orig_shape=orig_shape,
                            convrot_groupsize=self._convrot_groupsize,
                            quant_group_size=self._quant_group_size,
                            linear_dtype=self._linear_dtype,
                        )
                        
                        q_data = weight_tensor._qdata if isinstance(weight_tensor, QuantizedTensor) else weight_tensor
                        q_weight = QuantizedTensor(q_data, "TensorCoreConvRotW4A4Layout", params)
                        self.weight = nn.Parameter(q_weight, requires_grad=False)
                elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32) or is_fp8:
                    is_excluded = any(ex in prefix for ex in Int4Ops.excluded_names)
                    is_dim1 = self.in_features == 1 or self.out_features == 1 or weight_tensor.ndim == 1
                    should_quantize = not (is_excluded or is_dim1 or not Int4Ops.dynamic_quantize)
                    defer_finalize = (
                        getattr(Int4Ops, "dynamic_load_device", None) is not None
                        and (should_quantize or pending_weight_lora_patches)
                    )

                    if defer_finalize:
                        self._is_quantized = False
                        self.weight = nn.Parameter(source_tensor(weight_tensor), requires_grad=False)
                        self._pending_int4_finalize = {
                            "weight_key": weight_key,
                            "quantize": should_quantize,
                            "lora_patches": pending_weight_lora_patches,
                            "device": getattr(Int4Ops, "dynamic_load_device", None),
                            "convrot_groupsize": getattr(Int4Ops, "convrot_groupsize", 256),
                            "quant_group_size": getattr(Int4Ops, "quant_group_size", 64),
                            "linear_dtype": getattr(Int4Ops, "linear_dtype", "int4"),
                            "cast_fp8": is_fp8,
                        }
                        if pending_weight_lora_patches:
                            if not hasattr(Int4Ops, 'applied_lora_patches'):
                                Int4Ops.applied_lora_patches = set()
                            Int4Ops.applied_lora_patches.add(normalize_key(weight_key))
                    elif not should_quantize:
                        self._is_quantized = False
                        if is_fp8 and hasattr(weight_tensor, "to"):
                            weight_tensor = weight_tensor.to(dtype=torch.bfloat16)
                        self.weight = nn.Parameter(source_tensor(weight_tensor), requires_grad=False)
                    else:
                        device = getattr(Int4Ops, "dynamic_load_device", None)
                        if device is None:
                            device = torch.device("cuda") if torch.cuda.is_available() else weight_tensor.device
                        
                        if hasattr(weight_tensor, "to"):
                            w_gpu = weight_tensor.to(device=device).to(dtype=torch.float32)
                        else:
                            w_gpu = weight_tensor.to(device).float()
                        stochastic_rounding = 1 if getattr(Int4Ops, "lora_mode", "None") == "Stochastic" else 0
                        
                        qdata, params = TensorCoreConvRotW4A4Layout.quantize(
                            w_gpu,
                            convrot_groupsize=getattr(Int4Ops, "convrot_groupsize", 256),
                            quant_group_size=getattr(Int4Ops, "quant_group_size", 64),
                            stochastic_rounding=stochastic_rounding,
                            linear_dtype=getattr(Int4Ops, "linear_dtype", "int4"),
                        )
                        
                        params_cpu = TensorCoreConvRotW4A4Layout.Params(
                            scale=params.scale.cpu() if isinstance(params.scale, torch.Tensor) else params.scale,
                            orig_dtype=params.orig_dtype,
                            orig_shape=params.orig_shape,
                            convrot_groupsize=params.convrot_groupsize,
                            quant_group_size=params.quant_group_size,
                            linear_dtype=params.linear_dtype,
                        )
                        
                        self.weight = nn.Parameter(QuantizedTensor(qdata.cpu(), "TensorCoreConvRotW4A4Layout", params_cpu), requires_grad=False)
                        self._is_quantized = True
                        self._quant_format = "convrot_w4a4"
                        self._convrot_groupsize = params.convrot_groupsize
                        self._quant_group_size = params.quant_group_size
                        self._linear_dtype = params.linear_dtype
                        del w_gpu, qdata
                else:
                    self._is_quantized = False
                    self.weight = nn.Parameter(source_tensor(weight_tensor), requires_grad=False)
            else:
                missing_keys.append(weight_key)

            if bias_tensor is not None:
                self.bias = nn.Parameter(source_tensor(bias_tensor), requires_grad=False)
            else:
                self.bias = None

            if self.weight is not None:
                self.weight_comfy_model_dtype = self.weight.dtype
            if self.bias is not None:
                self.bias_comfy_model_dtype = self.bias.dtype

        def forward(self, x: Tensor) -> Tensor:
            if not self._is_quantized:
                return super().forward(x)

            if getattr(self, "_quant_format", "convrot_w4a4") == "int8_tensorwise":
                from . import int8_quant
                int8_quant._use_triton = getattr(Int8TensorwiseOps, "use_triton", True)

                need_cast = self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0
                if need_cast:
                    weight, bias, offload_stream = cast_bias_weight(
                        self, input=None, dtype=torch.int8, device=x.device,
                        bias_dtype=x.dtype, offloadable=True
                    )
                else:
                    weight = self.weight
                    bias = self.bias
                    offload_stream = None

                w_scale = self.weight_scale
                if isinstance(w_scale, torch.Tensor) and w_scale.device != x.device:
                    w_scale = w_scale.to(x.device, non_blocking=True)

                compute_dtype = getattr(Int4Ops, "compute_dtype", None)
                if compute_dtype is None:
                    compute_dtype = Int8TensorwiseOps._default_compute_dtype(x)

                x_shape = x.shape
                x_2d = x.reshape(-1, x_shape[-1])
                if x_2d.dtype != compute_dtype:
                    x_2d = x_2d.to(compute_dtype)

                if getattr(self, "_use_convrot", False):
                    from .convrot import build_hadamard, rotate_activation
                    group_size = getattr(self, "_convrot_groupsize", 256)
                    H = build_hadamard(group_size, device=x.device, dtype=x_2d.dtype)
                    x_2d = rotate_activation(x_2d, H, group_size=group_size)

                if x_2d.shape[0] > 16:
                    if getattr(self, "_is_per_row", False):
                        y = int8_forward_dynamic_per_row(x_2d, weight, w_scale, bias, compute_dtype)
                    else:
                        y = int8_forward_dynamic(x_2d, weight, w_scale, bias, compute_dtype)
                else:
                    w_float = dequantize_int8(weight, w_scale).to(x_2d.dtype)
                    bias_typed = bias.to(x_2d.dtype) if bias is not None else None
                    y = F.linear(x_2d, w_float, bias_typed)

                if self.lora_patches:
                    for lora_down, lora_up, lora_start, lora_size in self.lora_patches:
                        lD = lora_down.to(x.device, non_blocking=True)
                        lU = lora_up.to(x.device, non_blocking=True)
                        lora_x = F.linear(x_2d.to(lD.dtype), lD)
                        lora_y = F.linear(lora_x, lU)
                        if lora_start is not None:
                            y[:, lora_start:lora_start + lora_size] = (
                                y[:, lora_start:lora_start + lora_size] + lora_y.to(y.dtype)
                            )
                        else:
                            y = y + lora_y.to(y.dtype)

                if need_cast:
                    uncast_bias_weight(self, weight, bias, offload_stream)

                return y.reshape(*x_shape[:-1], y.shape[-1])

            else:
                y = super().forward(x)

                if self.lora_patches:
                    x_shape = x.shape
                    x_2d = x.reshape(-1, x_shape[-1])
                    for lora_down, lora_up, lora_start, lora_size in self.lora_patches:
                        lD = lora_down.to(x.device, non_blocking=True)
                        lU = lora_up.to(x.device, non_blocking=True)
                        lora_x = F.linear(x_2d.to(lD.dtype), lD)
                        lora_y = F.linear(lora_x, lU)
                        if lora_start is not None:
                            y[:, lora_start:lora_start + lora_size] = (
                                y[:, lora_start:lora_start + lora_size] + lora_y.to(y.dtype)
                            )
                        else:
                            y = y + lora_y.to(y.dtype)
                
                return y

    class GroupNorm(manual_cast.GroupNorm): pass
    class LayerNorm(manual_cast.LayerNorm): pass
    class Conv2d(manual_cast.Conv2d): pass
    class Conv3d(manual_cast.Conv3d): pass
    class ConvTranspose2d(manual_cast.ConvTranspose2d): pass
    class Embedding(manual_cast.Embedding): pass

    @classmethod
    def conv_nd(cls, dims, *args, **kwargs):
        if dims == 2: return cls.Conv2d(*args, **kwargs)
        elif dims == 3: return cls.Conv3d(*args, **kwargs)
        else: raise ValueError(f"unsupported dimensions: {dims}")


class INT4LowVramPatch:
    is_lowvram_patch = True

    def __init__(self, key, patches, module, lora_mode):
        self.key = key
        self.patches = patches
        self.module = module
        self.lora_mode = lora_mode
        self.prepared_patches = None

    def memory_required(self):
        if not _use_new_prefetch:
            return 0
        counter = [0]
        for patch in self.patches[self.key]:
            comfy.lora.prefetch_prepared_value(patch[1], counter, None, None, False)
        return counter[0]

    def prepare(self, *args, **kwargs):
        if _use_new_prefetch:
            destination = args[0] if len(args) > 0 else kwargs.get("destination")
            stream = args[1] if len(args) > 1 else kwargs.get("stream")
            copy = args[2] if len(args) > 2 else kwargs.get("copy", True)
            commit = args[3] if len(args) > 3 else kwargs.get("commit", True)

            counter = [0]
            prepared_patches = [
                (patch[0], comfy.lora.prefetch_prepared_value(patch[1], counter, destination, stream, copy), patch[2], patch[3], patch[4])
                for patch in self.patches[self.key]
            ]
            if commit:
                self.prepared_patches = prepared_patches
            return prepared_patches
        else:
            allocate_buffer = args[0] if len(args) > 0 else kwargs.get("allocate_buffer")
            stream = args[1] if len(args) > 1 else kwargs.get("stream")

            self.prepared_patches = [
                (patch[0], comfy.lora.prefetch_prepared_value(patch[1], allocate_buffer, stream), patch[2], patch[3], patch[4])
                for patch in self.patches[self.key]
            ]
            return self.prepared_patches

    def clear_prepared(self):
        self.prepared_patches = None

    def __call__(self, weight):
        patches = self.prepared_patches if self.prepared_patches is not None else self.patches[self.key]
        
        if getattr(self.module, "_quant_format", "convrot_w4a4") == "int8_tensorwise":
            scale = self.module.weight_scale
            if isinstance(scale, torch.Tensor):
                scale = scale.to(weight.device)
            weight_float = dequantize_int8(weight, scale)
            use_convrot = getattr(self.module, "_use_convrot", False)
            if use_convrot:
                group_size = getattr(self.module, "_convrot_groupsize", 256)
                try:
                    from .convrot import build_hadamard, rotate_weight
                    H = build_hadamard(group_size, device=weight.device, dtype=weight_float.dtype)
                    weight_float = rotate_weight(weight_float, H, group_size=group_size)
                except ImportError:
                    use_convrot = False
            
            patched_weight_float = comfy.lora.calculate_weight(patches, weight_float, self.key, intermediate_dtype=weight_float.dtype)
            if use_convrot:
                patched_weight_float = rotate_weight(patched_weight_float, H, group_size=group_size)
            
            from .int8_quant import quantize_int8, stochastic_round_int8_delta
            if self.lora_mode == "Stochastic":
                return stochastic_round_int8_delta(patched_weight_float, scale, seed=comfy.utils.string_to_seed(self.key))
            return quantize_int8(patched_weight_float, scale)
        else:
            weight_float = weight.dequantize()
            patched_weight_float = comfy.lora.calculate_weight(patches, weight_float, self.key, intermediate_dtype=weight_float.dtype)
            stochastic_rounding = 1 if self.lora_mode == "Stochastic" else 0
            qdata, params = TensorCoreConvRotW4A4Layout.quantize(
                patched_weight_float,
                convrot_groupsize=getattr(self.module, "_convrot_groupsize", 256),
                quant_group_size=getattr(self.module, "_quant_group_size", 64),
                stochastic_rounding=stochastic_rounding,
                linear_dtype=getattr(self.module, "_linear_dtype", "int4"),
            )
            return QuantizedTensor(qdata.to(weight.device), "TensorCoreConvRotW4A4Layout", params)


class INT4ModelPatcher(comfy.model_patcher.ModelPatcher):
    def finalize_pending_int8(self):
        finalized = 0
        for module in self.model.modules():
            finalize = getattr(module, "finalize_pending_int8", None)
            if finalize is not None and finalize():
                finalized += 1
        if finalized > 0:
            self.size = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        if key not in self.patches and not force_cast:
            return super().patch_weight_to_device(key, device_to, inplace_update, return_weight, force_cast)

        module_path = key.rsplit('.', 1)[0]
        try:
            module = comfy.utils.get_attr(self.model, module_path)
        except AttributeError:
            module = None

        is_int4_module = hasattr(module, "_is_quantized") and module._is_quantized
        patches = self.patches.get(key, [])

        if is_int4_module and Int4Ops.Linear._is_bias_key(key):
            return comfy.utils.get_attr(self.model, key) if return_weight else None

        if is_int4_module:
            if not Int4Ops.dynamic_lora:
                current_weight = comfy.utils.get_attr(self.model, key)

                if device_to is None:
                    device_to = current_weight.device

                if key not in self.backup:
                    import collections
                    BackupEntry = collections.namedtuple('Dimension', ['weight', 'inplace_update'])
                    self.backup[key] = BackupEntry(
                        weight=current_weight.to(device=self.offload_device, copy=inplace_update) if hasattr(current_weight, "to") else current_weight,
                        inplace_update=inplace_update,
                    )
                    source_weight = current_weight
                else:
                    source_weight = self.backup[key].weight

                if getattr(module, "_quant_format", "convrot_w4a4") == "int8_tensorwise":
                    scale = module.weight_scale
                    if isinstance(scale, torch.Tensor):
                        scale = scale.to(device_to)
                    weight_float = dequantize_int8(source_weight.to(device_to), scale)

                    use_convrot = getattr(module, "_use_convrot", False)
                    if use_convrot:
                        group_size = getattr(module, "_convrot_groupsize", 256)
                        try:
                            from .convrot import build_hadamard, rotate_weight
                            H = build_hadamard(group_size, device=device_to, dtype=weight_float.dtype)
                            weight_float = rotate_weight(weight_float, H, group_size=group_size)
                        except ImportError:
                            pass

                    patches_list = self.patches.get(key, [])
                    patched_weight_float = comfy.lora.calculate_weight(patches_list, weight_float, key)

                    if use_convrot:
                        patched_weight_float = rotate_weight(patched_weight_float, H, group_size=group_size)

                    from .int8_quant import quantize_int8, stochastic_round_int8_delta
                    if getattr(Int4Ops, "lora_mode", "None") == "Stochastic":
                        patched_weight_int8 = stochastic_round_int8_delta(patched_weight_float, scale)
                    else:
                        patched_weight_int8 = quantize_int8(patched_weight_float, scale)

                    patched_weight_int8 = patched_weight_int8.to(current_weight.device)

                    if return_weight:
                        return patched_weight_int8

                    if inplace_update:
                        current_weight.data.copy_(patched_weight_int8)
                    else:
                        comfy.utils.set_attr(self.model, key, nn.Parameter(patched_weight_int8, requires_grad=False))
                    return
                else:
                    weight_float = source_weight.dequantize().to(device_to)

                    patches_list = self.patches.get(key, [])
                    patched_weight_float = comfy.lora.calculate_weight(patches_list, weight_float, key)

                    stochastic_rounding = 1 if getattr(Int4Ops, "lora_mode", "None") == "Stochastic" else 0
                    qdata, params = TensorCoreConvRotW4A4Layout.quantize(
                        patched_weight_float,
                        convrot_groupsize=getattr(module, "_convrot_groupsize", 256),
                        quant_group_size=getattr(module, "_quant_group_size", 64),
                        stochastic_rounding=stochastic_rounding,
                        linear_dtype=getattr(module, "_linear_dtype", "int4"),
                    )

                    patched_weight_quant = QuantizedTensor(qdata.to(current_weight.device), "TensorCoreConvRotW4A4Layout", params)

                    if return_weight:
                        return patched_weight_quant

                    if inplace_update:
                        current_weight.data.copy_(patched_weight_quant)
                    else:
                        comfy.utils.set_attr(self.model, key, nn.Parameter(patched_weight_quant, requires_grad=False))
                    return
            else:
                weight = comfy.utils.get_attr(self.model, key)
                device = weight.device if weight is not None else self.offload_device
                lora_patches = []
                for p in patches:
                    strength_patch = p[0]
                    adapter = p[1]
                    strength_model = p[2]
                    offset = p[3] if len(p) > 3 else None

                    if not hasattr(adapter, "weights"):
                        continue

                    strength = strength_patch * strength_model
                    weights = adapter.weights
                    if len(weights) == 6:
                        up, down, alpha, mid, dora, reshape = weights
                        rank = down.shape[0] if down.ndim >= 2 else 1
                        scale = (alpha / rank) * strength if alpha is not None else strength

                        down_scaled = down.flatten(1) * scale
                        if mid is not None:
                            down_scaled = torch.mm(mid.flatten(1), down.flatten(1)) * scale

                        if getattr(module, "_convrot_groupsize", None) is not None and down_scaled.shape[1] % module._convrot_groupsize == 0:
                            try:
                                from .convrot import build_hadamard, rotate_weight
                                H = build_hadamard(module._convrot_groupsize, device=down_scaled.device, dtype=down_scaled.dtype)
                                down_scaled = rotate_weight(down_scaled, H, group_size=module._convrot_groupsize)
                            except ImportError:
                                pass

                        start, size = None, None
                        if offset is not None:
                            _dim, start, size = offset

                        lora_patches.append((down_scaled.to(device), up.flatten(1).to(device), start, size))

                module.lora_patches = lora_patches
                if return_weight:
                    return weight
                return
        
        return super().patch_weight_to_device(key, device_to, inplace_update, return_weight, force_cast)

    def load(self, *args, **kwargs):
        self.finalize_pending_int8()
        save_materialized = bool(getattr(self, "_int4_save_materialized_lora", False))

        if not Int4Ops.dynamic_lora and not save_materialized:
            for k in list(self.backup):
                if k in self.patches:
                    try:
                        module = comfy.utils.get_attr(self.model, k.rsplit('.', 1)[0])
                    except AttributeError:
                        module = None
                    if hasattr(module, "_is_quantized") and module._is_quantized:
                        bk = self.backup.pop(k)
                        if bk.inplace_update:
                            dest = comfy.utils.get_attr(self.model, k)
                            dest.data.copy_(bk.weight)
                        else:
                            comfy.utils.set_attr(self.model, k, bk.weight)

        stale_keys = [k for k in self.backup if k not in self.patches]
        for k in stale_keys:
            bk = self.backup.pop(k)
            if bk.inplace_update:
                dest = comfy.utils.get_attr(self.model, k)
                dest.data.copy_(bk.weight)
            else:
                comfy.utils.set_attr(self.model, k, bk.weight)

        for name, module in self.model.named_modules():
            if hasattr(module, "lora_patches") and module.lora_patches:
                if not Int4Ops.dynamic_lora or (name + ".weight") not in self.patches:
                    module.lora_patches = []

        res = super().load(*args, **kwargs) if hasattr(super(), "load") else None
        device_to = kwargs.get("device_to", args[0] if len(args) > 0 else self.model.device)

        for name, module in self.model.named_modules():
            if hasattr(module, "_is_quantized") and module._is_quantized:
                weight_key = name + ".weight"
                if weight_key in self.patches:
                    if save_materialized:
                        if hasattr(module, "weight_lowvram_function"):
                            module.weight_lowvram_function = None
                        if hasattr(module, "weight_function"):
                            module.weight_function = [f for f in getattr(module, "weight_function", []) if type(f).__name__ != "LowVramPatch"]
                    elif Int4Ops.dynamic_lora:
                        if hasattr(module, "weight_lowvram_function"):
                            module.weight_lowvram_function = None
                        if hasattr(module, "weight_function"):
                            module.weight_function = [f for f in getattr(module, "weight_function", []) if type(f).__name__ != "LowVramPatch"]
                        self.patch_weight_to_device(weight_key, device_to=device_to)
                    else:
                        if hasattr(module, "weight_function"):
                            module.weight_function = [f for f in getattr(module, "weight_function", []) if type(f).__name__ != "LowVramPatch"]
                        
                        lowvram_patch = INT4LowVramPatch(
                            weight_key,
                            self.patches,
                            module,
                            getattr(Int4Ops, "lora_mode", "None"),
                        )
                        pin_state = getattr(self.model, "dynamic_pins", {}).get(self.load_device, None)
                        if pin_state is not None:
                            lowvram_patch._pin_state = pin_state
                        module.weight_lowvram_function = lowvram_patch

        return res

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for name, module in self.model.named_modules():
                if hasattr(module, "lora_patches"):
                    module.lora_patches = []
        return super().unpatch_model(device_to, unpatch_weights)

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        if src_cls is INT4ModelPatcher:
            return super().clone(*args, **kwargs)
        if not issubclass(src_cls, INT4ModelPatcher):
            name = f"INT4_{src_cls.__name__}"
            dynamic_cls = type(name, (INT4ModelPatcher, src_cls), {})
        else:
            dynamic_cls = src_cls
        self.__class__ = dynamic_cls
        if not self.is_dynamic() and getattr(self, "cached_patcher_init", None) is None:
            self.cached_patcher_init = (lambda *a, **kw: self, ())
        n = super().clone(*args, **kwargs)
        disable_dyn = kwargs.get("disable_dynamic", False)
        if len(args) > 0:
            disable_dyn = args[0]
        if disable_dyn and not issubclass(n.__class__, INT4ModelPatcher):
            new_cls = type(f"INT4_{n.__class__.__name__}", (INT4ModelPatcher, n.__class__), {})
            n.__class__ = new_cls
        self.__class__ = src_cls
        return n
