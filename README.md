# Bo(dy) Mo(dels) To(ol)

A toolkit for fitting body models to meshes and converting between body models

## Installation

1. Clone the repository and its submodules:

    ```bash
    git clone --recurse-submodules https://github.com/NeelayS/bomoto.git
    ```

2. Install the PyTorch version of your choice, which is compatible with your GPU, from
   the [official website](https://pytorch.org/).

3. Install the dependencies for this repository:

    ```bash
    cd bomoto && pip install -e .
    ```
    
    Run these commands to install the SUPR and SKEL body models
    ```bash
    cd submodules/SUPR && pip install -e . && cd ../..
    cd submodules/SKEL && pip install -e . && cd ../..
    ```

## Usage

1. Depending on whether you input data is meshes or model parameter files, choose the appropriate config file from
   `configs/`.
2. Edit it to suit your needs.
3. Run the script:

    ```bash
    python run.py --cfg configs/<config_file>.yaml
    ```
