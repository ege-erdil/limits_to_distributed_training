# Limits to distributed training

This is the public repository for the paper "Limits to distributed training".

## How do I run this?

You can read through the Jupyter notebook "optimal_parallelism.ipynb". Running this notebook end-to-end will currently produce Figure 7 and Figure 8 from the paper, and the notebook explains which setting must be changed to reproduce the other plots and results as well.

## What are the other contents of the repository?

The "gpu" folder serves as a self-contained package that models matrix multiplications on a single GPU, and contains definitions of several popular NVIDIA GPUs such as the A100 and the H100. This module is imported by the main Jupyter notebook during execution.

The "logs" folder contains detailed outputs from all of our simulations: this contains much more detailed information than we were able to put inside the paper due to space constraints. The "visuals" folder contains the plots we have put in the body of the paper.
