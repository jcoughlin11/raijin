Goal:
    Easily run RL experiments with modular components. That is, I want to be
    able to easily mix and match things like network architecture, memory
    buffer type (e.g., sum tree, simple deque, etc), optimizers, loss functions,
    training loop structure, and data processing pipelines.

Research Question:
    Can the hdd footprint of the memory buffer be reduced by pooling similar
    experiences together without significantly reducing performance?
