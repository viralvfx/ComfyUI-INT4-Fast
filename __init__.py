import logging

# =============================================================================
# Layout Registration
# =============================================================================
def _register_layouts():
    try:
        from comfy.quant_ops import QUANT_ALGOS, QuantizedLayout
        
        # Register W8A8 (needed as fallback/mixed-precision in some models)
        if "TensorCoreConvRotW8A8Layout" not in QUANT_ALGOS:
            class TensorCoreConvRotW8A8Layout(QuantizedLayout):
                pass
            QUANT_ALGOS["TensorCoreConvRotW8A8Layout"] = TensorCoreConvRotW8A8Layout
            logging.info("Int4Fast: Registered TensorCoreConvRotW8A8Layout dynamically.")

        # Register W4A4
        if "TensorCoreConvRotW4A4Layout" not in QUANT_ALGOS:
            class TensorCoreConvRotW4A4Layout(QuantizedLayout):
                pass
            QUANT_ALGOS["TensorCoreConvRotW4A4Layout"] = TensorCoreConvRotW4A4Layout
            logging.info("Int4Fast: Registered TensorCoreConvRotW4A4Layout dynamically.")
    except Exception as e:
        logging.warning(f"Int4Fast: Failed to register layouts dynamically: {e}")

_register_layouts()

# =============================================================================
# Export Custom Ops
# =============================================================================
try:
    from .int4_quant import Int4Ops
except ImportError:
    Int4Ops = None

# =============================================================================
# Node Mappings
# =============================================================================
try:
    from .int4_unet_loader import UNetLoaderINTW4A4
    from .int4_save import INT4ModelSave
    
    NODE_CLASS_MAPPINGS = {
        "OTUNetLoaderW4A4": UNetLoaderINTW4A4,
        "INT4ModelSave": INT4ModelSave,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "OTUNetLoaderW4A4": "Load Diffusion Model INT4 (W4A4)",
        "INT4ModelSave": "Save Int4 Model",
    }
except ImportError as e:
    logging.error(f"Int4Fast: Failed to import nodes: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "Int4Ops",
]
