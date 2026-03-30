# Unified MLIP Interface

This directory provides a **unified Cuby4 interface for multiple MLIP backends** using a persistent worker process. Working [Cuby4](http://cuby4.molecular.cz/) software is required.

## Usage

```bash
cd $CUBY_DIR/interfaces/
git clone git@github.com:tobiasmarek/mlip-cuby-interface.git mlip # clone as `mlip`
conda env create -f mlip/envs/fairchem.yaml # e.g. for UMA models
conda activate fairchem
cuby4 /path/to/your/template.yaml # with `interface: mlip`
```

Example of `template.yaml`:
```yaml
# conda activate fairchem
job: dataset
dataset: PLA15

interface: mlip
method: mlip
mlip_backend: fairchem
mlip_model: uma-s-1p2
mlip_device: cpu
mlip_cpu_threads: 8
# more within `keywords.yaml`
```

## Supported backends

| Backend  | Models tested          | Status                      |
|----------|------------------------|-----------------------------|
| aimnet   | AIMNet2(2025)          | :white_check_mark: working  |
| fairchem | UMA models             | :white_check_mark: working  |
| fennol   | FeNNix-BIO1 models     | :white_check_mark: working  |
| mace     | MACE-POLAR-1 models    | :white_check_mark: working  |
| mlatom   | AIQM2, AIQM3           | :x: failing                 |
| nequip   | NequIP, Allegro models | :white_check_mark: working  |
| orbital  | ORB-V3 series          | :white_check_mark: working  |
| so3lr    | SO3LR                  | :warning: untested          |
| torchmd  | PM6-ML                 | :white_check_mark: working  |

## Files

- `mlip.rb` – main Ruby interface
- `mlip_workers.py` – definitions of MLIP backends
- `mlip_worker_server.py` – persistent Python worker server
- `mlip_bridge_client.py` – bridge between Ruby and the worker transport
- `mlip_client.py` – Python transport client used by the bridge
- `keywords.yaml` – interface keywords

## How it works

The model is loaded once and kept alive in a Python worker process.

Ruby (`mlip.rb`) acts only as a manager:

1. Start worker server.
2. Start bridge process.
3. Send structures for repeated calculations.
4. Read results and convert them to Cuby `Results`.
5. Shut everything down cleanly.

## Adding a new backend

1) Within [`mlip_workers.py`](mlip_workers.py) define a new child class of the MLIPWorker abstract class

2) Implement the *load* method which handles the loading of the model and preparing it for inference

3) Implement the *calculate* method which takes an XYZ string, a gradients flag, and an optional charge, and returns a dictionary with "energy" and optionally "forces"

4) Use the *resolve_torch_device* and *apply_torch_limits* static methods from MLIPWorker for PyTorch-based workers to handle device selection and resource limits

5) Ensure that all workers return energy in *kcal/mol* and forces in *kcal/mol/Å* for consistency, use ase.units for unit conversions if needed

6) Add import to [`mlip_worker_server.py`](mlip_worker_server.py) and register the worker class in the WORKER_CLASSES dictionary

7) Add the worker name to keywords.yaml