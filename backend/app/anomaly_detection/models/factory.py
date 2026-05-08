from .glc_flow_v7 import GLCFlowV7
from ..schemas import GLCFlowConfig


def create_glc_flow_v7(config: GLCFlowConfig | None = None) -> GLCFlowV7:
    """Factory function to create a GLCFlowV7 model instance.

    Args:
        config: GLCFlowConfig with model hyperparameters.
                If None, uses default configuration.

    Returns:
        GLCFlowV7 model instance ready for training.
    """
    if config is None:
        config = GLCFlowConfig()
    return GLCFlowV7(config)


def count_parameters(model: GLCFlowV7) -> int:
    """Count total trainable parameters in GLCFlowV7.

    Args:
        model: GLCFlowV7 instance

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: GLCFlowV7) -> dict:
    """Get a summary of model architecture and parameter counts.

    Args:
        model: GLCFlowV7 instance

    Returns:
        dict with component-wise parameter counts and total
    """
    components = {
        "global_revin": model.global_revin,
        "local_revin": model.local_revin,
        "patchify": model.patchify,
        "flatten": model.flatten,
        "encoder_global": model.encoder_global,
        "encoder_local": model.encoder_local,
        "pos_embedding": model.pos_embedding,
        "cls_token": model.cls_token,
        "transformer": model.transformer,
        "pooling": model.pooling,
        "global_head": model.global_head,
        "local_head": model.local_head,
        "cnf": model.cnf,
    }

    param_counts = {}
    for name, comp in components.items():
        param_counts[name] = sum(p.numel() for p in comp.parameters() if p.requires_grad)

    total = sum(param_counts.values())

    return {
        "total_params": total,
        "trainable_params": total,
        "by_component": param_counts,
    }