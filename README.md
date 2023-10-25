# (Bo)dy (Mo)dels (To)ol

A toolkit for fitting body models to meshes and converting between body models

## Installation

1. Clone the repository and its submodules:

    ```bash
    git clone --recurse-submodules https://github.com/NeelayS/bomoto.git
    ```

2. Install the PyTorch version of your choice, which is compatible with your GPU, from the [official website](https://pytorch.org/).

3. Install the dependencies for this repository:

    ```bash
    cd bomoto && python setup.py install
    cd SUPR && python setup.py install && cd ..
    ```

## Usage