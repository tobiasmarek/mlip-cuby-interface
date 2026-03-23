# Unified MLIP Interface

This directory contains a unified Cuby4 interface for multiple MLIP backends with a persistent worker process.

## Usage

```bash
cd $CUBY_DIR/interfaces/
git clone git@github.com:tobiasmarek/mlip-cuby-interface.git mlip # clone as `mlip` directory
conda env create -f mlip/envs/fairchem.yaml # for UMA models
cuby4 template.yaml
```

## What Was Built

- A single Ruby interface: `mlip.rb`
- A persistent Python worker server: `mlip_worker_server.py`
- A persistent Python bridge between Ruby and worker transport: `mlip_bridge_client.py`
- A Python transport client module (used by the bridge): `mlip_client.py`
- Interface keywords: `keywords.yaml`

## Main Idea

The model is loaded once and kept alive in a Python worker process.

Ruby (`mlip.rb`) acts only as a manager:

1. Start worker server.
2. Start bridge process.
3. Send structures for repeated calculations.
4. Read results and convert them to Cuby `Results`.
5. Shut everything down cleanly.

## Backends

Supported backend names:

- `torchmd`
- `fairchem`
- `fennol`

Backend selection:

- Explicit: `mlip_model: "torchmd::/path/model.ckpt"`
- Automatic by extension:
  - `.fnx` -> `fennol`
  - `.pt` -> `fairchem`
  - otherwise -> `torchmd`

## Keywords

Defined in `keywords.yaml`:

- `mlip_model`
- `mlip_device`
- `mlip_cpu_threads`
- `mlip_cuda_memory_fraction`
- `mlip_multiplier`
- `mlip_sp_only`
- `mlip_set_atom_to_zero`
