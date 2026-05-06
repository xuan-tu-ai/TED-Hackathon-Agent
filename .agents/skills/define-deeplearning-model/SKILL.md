---
name: define-deeplearning-model
description: Standard practices for defining PyTorch deep learning models.
---

# Define Deep Learning Model Skill

When creating or modifying a deep learning model in PyTorch, you MUST adhere to the following strict rules:

1. **Strict Initialization Typing**: `__init__` arguments must have clear typing. Do not use `Any`.
2. **Dictionary Input/Output**: The input and output of the `forward` method must be a `dict` (of tensors). Do not use tuple or positional arguments.

## Reference Template
Refer to the following template for an example of how to structure your model: 
[_pytorch_model.py](./references/_pytorch_model.py)
