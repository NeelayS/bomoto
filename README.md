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
    cd submodules/SUPR && pip install . && cd ../..
    cd submodules/SKEL && pip install . && cd ../..
    ```

## Usage

1. Depending on whether you input data is meshes or model parameter files, choose the appropriate config file from
`configs/`.
2. Edit it to suit your needs.
3. Run the script:

    ```bash
    python run.py --cfg configs/<config_file>.yaml
    ```

## Examples
### Converting parameters
Convert parameters from one body model to another.
##### SMPL to SMPL-X
- Modify `examples/smpl2smplx/cfg.yaml`. In particular:
  - replace `input.body_model.path` with the path to your SMPL neutral model
  - replace `output.body_model.path` with the path to your SMPL-X neutral model
- Run the following commands
   ```bash
   python examples/smpl2smplx/generate_sample_data.py
   python run.py --cfg examples/smpl2smplx/cfg.yaml
   ```
- Check the results (SMPL-X parameters and meshes in obj format) in `examples/smpl2smplx/results` 

### Fitting parameters to meshes (aka `parms_for`)
Given a set of meshes, fit body model parameters to them.
##### Meshes to SMPL-X
- Modify `examples/smpl2smplx/cfg.yaml`. In particular:
  - replace `output.body_model.path` with the path to your SMPL-X neutral model
- Run the following commands
   ```bash
   python examples/parms_for_smplx/generate_sample_data.py --model_path <path to your SMPL-X neutral model npz file>
   python run.py --cfg examples/parms_for_smplx/cfg.yaml
   ```
- Check the results (SMPL-X parameters and meshes in obj format) in `examples/parms_for_smplx/results`
