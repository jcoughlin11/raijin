Goal:
    Easily run RL experiments with modular components. That is, I want to be able to easily mix and match things like network architecture, memory buffer type (e.g., sum tree, simple deque, etc), optimizers, loss functions, training loop structure, and data processing pipelines.

Research Question:
    Can the hdd footprint of the memory buffer be reduced by pooling similar
    experiences together without significantly reducing performance?

Tools:
    * clearML: https://allegro.ai/clearml/docs/index.html
    * pytorch lightning: https://pytorch-lightning.readthedocs.io/en/latest/
    * omegaconf: https://omegaconf.readthedocs.io/en/2.0_branch/usage.html

Example:
    * https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=4ARIT37rDdIZ

* Lightning uses modules (systems) and models.
    * A model is a single network
    * A module defines how a collection of models interact with one another
    * Modules should be self-contained


* Structure:
    run: The code's main function. It:
        * Parses command-line arguments
        * Reads the parameter file
        * Creates the desired system
        * Performs the desired operations on the system (e.g., training)

    system: Contains all core research ingredients:
        * Model
        * Optimizer
        * Train/test/validation steps

    Model:
        * Defines layers
        * Defines forward pass through network

* Use DataModules (pipelines). They can be passed to trainer.fit(module, dataModule)


* Ingredients:
    * env
    * memory
    * explore-exploit strategy
    * networks
    * loss functions
    * optimizers
    * training step
    * data pipeline
