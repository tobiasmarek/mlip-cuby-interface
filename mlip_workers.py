"""A collection of MLIP worker implementations for different backends."""
from __future__ import annotations

import abc
import io
import faulthandler, sys
faulthandler.enable(file=sys.stderr, all_threads=True)

from typing import Any, Dict, Optional, Tuple


class MLIPWorker(abc.ABC):
    """Abstract model worker API shared by all backends."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        sp_only: bool = False,
        cpu_threads: int = 0,
        cuda_memory_fraction: Optional[float] = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.sp_only = sp_only
        self.cpu_threads = cpu_threads
        self.cuda_memory_fraction = cuda_memory_fraction
        # TODO: Add unit conversion here?
        # TODO: Add restrictions on allowed elements here?
        # TODO: Add versions to each worker
        # Notes:
        # - Removed: charge: Optional[int] = None to charge: int
        # - Removed: int(charge) to just charge (because we suppose charge: int)

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights into memory and device."""

    @abc.abstractmethod
    def calculate(self, xyz: str, gradients: bool, charge: int) -> Dict[str, Any]:
        """Calculate energy and optional forces for one XYZ structure."""


class TorchBackedMLIPWorker(MLIPWorker):
    """Shared runtime setup for workers that use PyTorch directly."""

    @staticmethod
    def resolve_torch_device(torch_module: Any, requested: str) -> str:
        """Resolve the runtime device for a PyTorch-based worker."""
        requested = (requested or "auto").strip().lower()
        if requested == "auto":
            return "cuda:0" if torch_module.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch_module.cuda.is_available():
            return "cpu"
        if requested == "cuda":
            return "cuda:0"
        return requested

    @staticmethod
    def apply_torch_limits(torch_module: Any, cpu_threads: int, cuda_memory_fraction: Optional[float], runtime_device: str) -> None:
        """Apply resource limits for PyTorch-based workers."""
        if cpu_threads > 0:
            torch_module.set_num_threads(cpu_threads)
            if hasattr(torch_module, "set_num_interop_threads"):
                try:
                    torch_module.set_num_interop_threads(cpu_threads)
                except RuntimeError:
                    pass

        if cuda_memory_fraction is not None and runtime_device.startswith("cuda") and torch_module.cuda.is_available():
            device_index = 0
            if ":" in runtime_device:
                try:
                    device_index = int(runtime_device.split(":", 1)[1])
                except ValueError:
                    device_index = 0
            torch_module.cuda.set_per_process_memory_fraction(cuda_memory_fraction, device=device_index)

    def setup_torch_runtime(self, torch_module: Any) -> str:
        """Resolve device and apply PyTorch-specific resource limits."""
        runtime_device = self.resolve_torch_device(torch_module=torch_module, requested=self.device)
        self.apply_torch_limits(
            torch_module=torch_module,
            cpu_threads=self.cpu_threads,
            cuda_memory_fraction=self.cuda_memory_fraction,
            runtime_device=runtime_device,
        )
        self._torch = torch_module
        return runtime_device

    @staticmethod
    def torch_calculator_device(runtime_device: str) -> str:
        """Return the device form expected by ASE-style torch calculators."""
        return "cuda" if runtime_device.startswith("cuda") else "cpu"


################################################################################
#
# TORCHMDNET worker
#
# Status: Works (not tested with gradients and gpu)
#
# Notes: WARNING: Keyword "charge" not found in the input, using default value "0".
#
################################################################################

class TorchMDNetWorker(TorchBackedMLIPWorker):
    KJ_TO_KCAL = 1.0 / 4.184
    ATOMTYPES = {
        "Br": 1,
        "C": 3,
        "Ca": 5,
        "Cl": 7,
        "F": 9,
        "H": 10,
        "I": 12,
        "K": 13,
        "Li": 14,
        "Mg": 15,
        "N": 17,
        "Na": 19,
        "O": 21,
        "P": 23,
        "S": 26,
    }

    def load(self) -> None:
        import torch
        from torchmdnet.models.model import load_model

        runtime_device = self.setup_torch_runtime(torch)
        self._runtime_device = torch.device(runtime_device)
        self._model = load_model(self.model_path, derivative=not self.sp_only)
        self._model = self._model.to(self._runtime_device)

    @staticmethod
    def _parse_xyz(xyz: str) -> Tuple[list[str], list[list[float]]]:
        lines = xyz.strip().splitlines()
        if len(lines) < 2:
            raise ValueError("Invalid XYZ payload")

        natoms = int(lines[0].strip())
        atom_lines = lines[2 : 2 + natoms]
        if len(atom_lines) != natoms:
            raise ValueError("XYZ atom count mismatch")

        symbols = []
        coords = []
        for line in atom_lines:
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid XYZ atom line: {line}")
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return symbols, coords

    def calculate(self, xyz: str, gradients: bool, charge: int) -> Dict[str, Any]:
        # Read molecule
        symbols, coords = self._parse_xyz(xyz)
        try:
            atomtypes = [self.ATOMTYPES[symbol] for symbol in symbols]
        except KeyError as exc:
            raise ValueError(f"Element '{exc.args[0]}' not supported by TorchMD backend") from exc

        types = self._torch.tensor(atomtypes, dtype=self._torch.long, device=self._runtime_device)
        positions = self._torch.tensor(coords, dtype=self._torch.float32, device=self._runtime_device)

        # Calculate
        result = self._model.forward(types, positions)
        if isinstance(result, tuple):
            energy = result[0]
            forces = result[1]
        else:
            energy = result
            forces = None

        energy_kcal = float(energy.item()) * self.KJ_TO_KCAL
        payload: Dict[str, Any] = {"energy": energy_kcal, "forces": None}

        if gradients:
            if forces is None:
                raise RuntimeError("Model loaded without derivatives; gradients are not available")
            payload["forces"] = (forces.detach().cpu().numpy() * self.KJ_TO_KCAL).tolist()

        return payload


################################################################################
#
# AIMNET2 worker for Aimnet2 models
#
# Status: Works (not tested with gradients and gpu)
#
# Notes:
# - TODO: Add support for electrostatic switches
# - We have to reinitialize the calculator for each structure (their bug)
# - UserWarning: State dict mismatch during model loading. Unexpected keys: ['outputs.dipole.mass', 'outputs.quadrupole.mass']
#   self.model, metadata = load_model(p, device=self.device)
#
################################################################################

class AimnetWorker(TorchBackedMLIPWorker):
    def load(self) -> None:
        import ase.units
        import torch

        torch_device = self.setup_torch_runtime(torch)

        self._electrostatics = None # "dsf" or "ewald" or None to disable long-range electrostatics
        self._runtime_device = self.torch_calculator_device(torch_device)
        self._aimnet_predict_eager = torch.compiler.disable(recursive=True, reason="AIMNet eager-only workaround")(self._aimnet_predict)
        self._ev_to_kcal = ase.units.mol / ase.units.kcal

    def _get_predictor(self, model_name: str) -> Any:
        from aimnet.calculators import AIMNet2Calculator

        calc = AIMNet2Calculator(
            model=model_name,
            device=self._runtime_device,
            compile_model=False
        )
        self._configure_electrostatics(calc, self._electrostatics)

        return calc

    def _configure_electrostatics(self, calc: Any, config: Optional[str]) -> None:
        if config is None:
            return
        if config.lower() == "dsf":
            # Damped-Shifted Force (DSF) - recommended for periodic systems
            calc.set_lrcoulomb_method("dsf", cutoff=15.0, dsf_alpha=0.2)
        if config.lower() == "ewald":
            # Ewald summation - for accurate periodic electrostatics
            calc.set_lrcoulomb_method("ewald", ewald_accuracy=1e-8)

    def _aimnet_predict(self, calc, data, gradients):
        return calc(data, forces=gradients, stress=False, hessian=False)

    def calculate(self, xyz: str, gradients: bool, charge: int) -> Dict[str, Any]:
        import ase.io
        import numpy as np

        # Read molecule
        atoms = ase.io.read(io.StringIO(xyz), format="xyz", index=0)

        # Set data and charge
        data = {
            "coord": np.asarray(atoms.positions, dtype=np.float64),
            "numbers": np.asarray(atoms.numbers, dtype=np.int64),
            "charge": float(charge),
        }

        # Add calculator
        calc = self._get_predictor(self.model_path)

        # Calculate
        with self._torch.compiler.set_stance("force_eager"):
            results = self._aimnet_predict_eager(calc, data, gradients)

        energy_kcal = float(results["energy"] * self._ev_to_kcal)
        payload: Dict[str, Any] = {"energy": energy_kcal, "forces": None}

        if gradients:
            forces = results["forces"] * self._ev_to_kcal
            payload["forces"] = forces.tolist()

        return payload


################################################################################
#
# FAIRCHEM worker for UMA models
#
# Status: Works (not tested with gradients and gpu)
#  - FIXME: doesn't work with zmq, maybe due to the fact that zmq handles
#    limiting cpu threads badly, or charge cache handling
#
# Notes:
#  - WARNING:root:If 'dataset_list' is provided in the config, the code
#   assumes that each dataset maps to itself. Please use 'dataset_mapping' as
#  'dataset_list' is deprecated and will be removed in the future.
#  - Needs HuggingFace token to get the models (`hf auth login`)
#
################################################################################

class FairchemWorker(TorchBackedMLIPWorker):
    def load(self) -> None:
        import ase.units
        import torch
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        torch_device = self.setup_torch_runtime(torch)
        self._runtime_device = self.torch_calculator_device(torch_device)
        self._predictor = pretrained_mlip.load_predict_unit(self.model_path, device=self._runtime_device)
        self._calculator_cls = FAIRChemCalculator
        self._ev_to_kcal = ase.units.mol / ase.units.kcal

    def calculate(self, xyz: str, gradients: bool, charge: int) -> Dict[str, Any]:
        import ase.io

        # Read molecule
        atoms = ase.io.read(io.StringIO(xyz), format="xyz", index=0)

        # Set charge
        atoms.info.update({"charge": charge, "spin": 1})

        # Add calculator
        atoms.calc = self._calculator_cls(self._predictor, task_name="omol")

        # Calculate
        energy_kcal = float(atoms.get_potential_energy() * self._ev_to_kcal)
        payload: Dict[str, Any] = {"energy": energy_kcal, "forces": None}

        if gradients:
            forces = atoms.get_forces() * self._ev_to_kcal
            payload["forces"] = forces.tolist()

        return payload


################################################################################
#
# FENNOL worker for FeNNix models
#
# Status: Works (not tested with gradients and gpu)
#
# Notes: FENNIXCalculator was modified to handle total charge on input
#
################################################################################

class FennolWorker(MLIPWorker):
    def load(self) -> None:
        import ase.units
        from fennol.ase import FENNIXCalculator

        self._calculator_cls = FENNIXCalculator
        self._predictor_cache: Dict[Optional[int], Any] = {}
        self._ev_to_kcal = ase.units.mol / ase.units.kcal

    def _predictor(self, charge: Optional[int]) -> Any:
        if charge not in self._predictor_cache:
            self._predictor_cache[charge] = self._calculator_cls(
                model=self.model_path,
                verbose=False,
                total_charge=charge,
            )
        return self._predictor_cache[charge]

    def calculate(self, xyz: str, gradients: bool, charge: int) -> Dict[str, Any]:
        import ase.io

        # Read molecule
        atoms = ase.io.read(io.StringIO(xyz), format="xyz", index=0)

        # Set charge and calculator
        atoms.calc = self._predictor(charge)

        # Calculate
        energy_kcal = float(atoms.get_potential_energy() * self._ev_to_kcal)
        payload: Dict[str, Any] = {"energy": energy_kcal, "forces": None}

        if gradients:
            forces = atoms.get_forces() * self._ev_to_kcal
            payload["forces"] = forces.tolist()

        return payload
    

################################################################################
#
# MACE worker for MACE models
#
# Status: Works (not tested with gradients and gpu)
#
# Notes:
# - TODO: Add precision flag / or kwargs for all worker types to handle this in a more generic way
# - TODO: Add support for -anicc MACE models
# - Very memory hungry due to cluster expansion (ig)
# - Higher number of CPUs and memory recommended
#
################################################################################

class MACEWorker(TorchBackedMLIPWorker):
    def load(self) -> None:
        import ase.units
        import torch

        torch_device = self.setup_torch_runtime(torch)
        self._runtime_device = self.torch_calculator_device(torch_device)
        self._predictor = self._get_predictor(self.model_path)
        self._ev_to_kcal = ase.units.mol / ase.units.kcal

    def _get_predictor(self, model_name: str) -> Any:
        import mace
        from mace.calculators import MACECalculator
        from mace.calculators import mace_polar, mace_off, mace_anicc, mace_omol, mace_mp

        if model_name.endswith(".model"): # when downloaded .model files are used
            return MACECalculator(model_paths=model_name, device=self._runtime_device)
        elif "polar" in model_name:
            return mace_polar(model=model_name, device=self._runtime_device) #, return_raw_model=True, default_dtype=self.precision)
        elif "off" in model_name:
            return mace_off(model=model_name, device=self._runtime_device)
        elif "anicc" in model_name:
            return mace_anicc(model=model_name, device=self._runtime_device)
        elif "omol" in model_name: # extra_large by default
            return mace_omol(model="extra_large", device=self._runtime_device)
        else:
            raise ValueError(f"Model name {model_name} does not match any known MACE model type")

    def calculate(self, xyz: str, gradients: bool, charge: int) -> Dict[str, Any]:
        import ase.io

        # Read molecule
        atoms = ase.io.read(io.StringIO(xyz), format="xyz", index=0)

        # Set charge
        atoms.info.update({"charge": charge, "spin": 1}) #, "external_field": [0.0, 0.0, 0.0]})

        # Add calculator
        atoms.calc = self._predictor

        # Calculate
        energy_kcal = float(atoms.get_potential_energy() * self._ev_to_kcal)
        payload: Dict[str, Any] = {"energy": energy_kcal, "forces": None}

        if gradients:
            forces = atoms.get_forces() * self._ev_to_kcal
            payload["forces"] = forces.tolist()

        return payload
    

################################################################################
#
# ORBITAL worker for Orbital-v3 models
#
# Status: Works (not tested with gradients and gpu)
#
# Notes:
# - FIXME: Doesn't work with zmq
#
# If you have several graphs, batch them like so:
# graph = atoms_adapter.batch([graph1, graph2])
# or 
# graph = atoms_adapter.from_ase_atoms_list([atoms1, atoms2])
#
################################################################################

class OrbitalWorker(TorchBackedMLIPWorker):
    def load(self) -> None:
        import ase.units
        import torch
        from orb_models.forcefield.inference.calculator import ORBCalculator
        from orb_models.common.utils import seed_everything

        self._runtime_device = self.setup_torch_runtime(torch)
        self._predictor, self._atoms_adapter = self._get_predictor(self.model_path, precision="float32-high") # or "float32-highest" / "float64 
        seed_everything(42)
        self._calculator_cls = ORBCalculator
        self._ev_to_kcal = ase.units.mol / ase.units.kcal

    def _get_predictor(self, model_name: str, precision: str) -> Any:
        from orb_models.forcefield import pretrained

        if model_name == "orb-v3-conservative-omol":
            return pretrained.orb_v3_conservative_omol(device=self._runtime_device, precision=precision)
        elif model_name == "orb-v3-direct-omol":
            return pretrained.orb_v3_direct_omol(device=self._runtime_device, precision=precision)
        else:
            # Fallback for materials if you really intended to use them, but warn the user
            print(f"Warning: Loading generic/material model {model_name}. Charge might be ignored.")
            # Try to load it dynamically if it exists in pretrained
            if hasattr(pretrained, model_name.replace("-", "_")):
                return getattr(pretrained, model_name.replace("-", "_"))(device=self._runtime_device, precision=precision)
            else:
                raise ValueError(f"Model {model_name} not found.")

    def calculate(self, xyz: str, gradients: bool, charge: int) -> Dict[str, Any]:
        import ase.io

        # Read molecule
        atoms = ase.io.read(io.StringIO(xyz), format="xyz", index=0)

        # Set charge
        atoms.info.update({"charge": charge, "spin": 1}) #, "external_field": [0.0, 0.0, 0.0]})

        # Add calculator
        atoms.calc = self._calculator_cls(self._predictor, atoms_adapter=self._atoms_adapter, device=self._runtime_device)

        # Calculate
        energy_kcal = float(atoms.get_potential_energy() * self._ev_to_kcal)
        payload: Dict[str, Any] = {"energy": energy_kcal, "forces": None}

        if gradients:
            forces = atoms.get_forces() * self._ev_to_kcal
            payload["forces"] = forces.tolist()

        return payload
    

################################################################################
#
# MLATOM worker for AIQM models
#
# Status: Fails (not tested with gradients and gpu)
#
# Notes:
# - FIXME: Fails to converge for some structures thx to semiempirics
# - Needs Aitomic addon for AIQM3 access
#
################################################################################

class MlatomWorker(MLIPWorker):
    def load(self) -> None:
        import ase.units
        import aitomic as ml

        self._mlip_module = ml
        self._predictor_initialized = False
        self._predictor = self._get_predictor(self.model_path)
        self._ev_to_kcal = ase.units.mol / ase.units.kcal

    def _get_predictor(self, model_name: str, atoms: str = None) -> Any:
        if model_name == "uaiqm_optimal":
            if atoms is None:
                return
            predictor = self._mlip_module.models.uaiqm(method=model_name, verbose=False)
            predictor.warning=False # Suppress warnings
            predictor.select_optimal(molecule=atoms) #,nCPUs=1,time_budget='1min')
            self._predictor_initialized = True
        else:
            predictor = self._mlip_module.models.methods(method=model_name, baseline_kwargs={'etemp': 400}) # for PLA15 dataset (with big structs) 400 works without displace
            predictor.warning=False # Suppress warnings
            self._predictor_initialized = True
        return predictor

    def calculate(self, xyz: str, gradients: bool, charge: int) -> Dict[str, Any]:
        # Read molecule
        atoms = self._mlip_module.data.molecule()
        atoms = atoms.read_from_xyz_string(xyz)

        # Set charge
        atoms.charge=charge
        atoms.spin = 0
        atoms.multiplicity = 1

        # Add calculator
        if not self._predictor_initialized:
            self._predictor = self._get_predictor(self.model_path, atoms=atoms)

        # Calculate
        self._predictor.predict(molecule=atoms, calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False)
        energy_kcal = float(atoms.energy * self._ev_to_kcal)
        payload: Dict[str, Any] = {"energy": energy_kcal, "forces": None}

        if gradients:
            self._predictor.predict(molecule=atoms, calculate_energy=False, calculate_energy_gradients=True, calculate_hessian=False)
            forces = atoms.forces * self._ev_to_kcal # FIXME: Or atoms.gradients?
            payload["forces"] = forces.tolist()

        return payload
    

################################################################################
#
# SO3LR worker
#
# Status: Untested
#
# Notes:
#
#
################################################################################

class So3lrWorker(MLIPWorker):
    def load(self) -> None:
        import ase.units
        import numpy as np

        self._predictor = self._get_predictor(None, precision=np.float64)
        self._ev_to_kcal = ase.units.mol / ase.units.kcal

    def _get_predictor(self, model_name: Optional[str], precision: Any) -> Any:
        from so3lr import So3lrCalculator

        return So3lrCalculator(
            calculate_stress=False,
            dtype=precision,
            lr_cutoff=1000.0,
            dispersion_energy_cutoff_lr_damping = 2.0
        )

    def calculate(self, xyz: str, gradients: bool, charge: int) -> Dict[str, Any]:
        import ase.io

        # Read molecule
        atoms = ase.io.read(io.StringIO(xyz), format="xyz", index=0)

        # Set charge
        atoms.info.update({"charge": charge, "spin": 1})
        
        # Add calculator
        atoms.calc = self._predictor

        # Calculate
        energy_kcal = float(atoms.get_potential_energy() * self._ev_to_kcal)
        payload: Dict[str, Any] = {"energy": energy_kcal, "forces": None}

        if gradients:
            forces = atoms.get_forces() * self._ev_to_kcal
            payload["forces"] = forces.tolist()

        return payload


################################################################################
#
# NEQIUP worker for NequIP and Allegro models
#
# Status: Works (not tested with gradients)
#
# Notes:
#
#
################################################################################

class NequipWorker(TorchBackedMLIPWorker):
    def load(self) -> None:
        import ase.units
        import torch

        self._runtime_device = self.setup_torch_runtime(torch)

        import cuequivariance_torch
        from nequip.integrations.ase import NequIPCalculator

        self._calculator_cls = NequIPCalculator
        self._predictor = self._get_predictor(self.model_path)
        self._kj_to_kcal = ase.units.kJ / ase.units.kcal
        self._ev_to_kcal = ase.units.mol / ase.units.kcal

    def _get_predictor(self, model_name: str) -> Any:
        return self._calculator_cls.from_compiled_model(
            compile_path=model_name,
            device=self._runtime_device,
            chemical_species_to_atom_type_map=True  # identity mapping (or mapping e.g. {"H": "H+", "C": "C_sp3", "O": "O-"})
        )

    def calculate(self, xyz: str, gradients: bool, charge: int) -> Dict[str, Any]:
        import ase.io

        # Read molecule
        atoms = ase.io.read(io.StringIO(xyz), format="xyz", index=0)

        # Set charge (Allegro is charge agnostic)
        atoms.info.update({"charge": charge, "spin": 1})
        
        # Add calculator
        atoms.calc = self._predictor

        # Calculate
        if "mir-group" in self.model_path: # official NequIP/Allegro models are in eV
            energy_kcal = float(atoms.get_potential_energy() * self._ev_to_kcal)
        else:
            energy_kcal = float(atoms.get_potential_energy() * self._kj_to_kcal)
        payload: Dict[str, Any] = {"energy": energy_kcal, "forces": None}

        if gradients:
            if "mir-group" in self.model_path:
                forces = atoms.get_forces() * self._ev_to_kcal
            else:
                forces = atoms.get_forces() * self._kj_to_kcal
            payload["forces"] = forces.tolist()

        return payload


################################################################################
#
# UBIO worker for UBio-MolFM models
#
# Status: Works (not tested with gradients and gpu)
#
# Notes:
#   - Config of the model should be in the same folder as model weights named "config.yaml"
#   - the config is static - if triton with cuda is used, need to modify config
#   - Needed to create a conda var of molfm since it is not on pip
#   - Uses less cpus even if given more
#   - Periodic boundaries in example calculations (here unset)
#   - Charge agnostic model, trained ONLY on NEUTRAL data
#
################################################################################

class UBioWorker(TorchBackedMLIPWorker):
    def load(self) -> None:
        import ase.units
        import torch

        self._runtime_device = self.setup_torch_runtime(torch)

        from molfm.interface.ase.calculator.e2former_calculator import E2FormerCalculator

        self._calculator_cls = E2FormerCalculator
        self._predictor = self._get_predictor(self.model_path)
        self._ev_to_kcal = ase.units.mol / ase.units.kcal

    def _get_predictor(self, model_name: str) -> Any:
        return self._calculator_cls(
            checkpoint_path=model_name, # "ubio-molfm-v1.5/molfm-v1p5-stage-3.pt", # molfm-v1p5-stage-3.pt, molfm-v1-stage-3.pt
            config_name="config.yaml",
            head_name="omol25",
            device=self._runtime_device,
            use_faiss=False,
            use_tf32=False, # TODO
            use_compile=False,
        )

    def calculate(self, xyz: str, gradients: bool, charge: int) -> Dict[str, Any]:
        import ase.io

        # Read molecule
        atoms = ase.io.read(io.StringIO(xyz), format="xyz", index=0)
        # atoms.set_cell([50.0, 50.0, 50.0])
        # atoms.center(vacuum=15) # vacuum = 15
        # atoms.pbc = [False, False, False] # periodic boundary conditions (defaultly [True, True, True])

        # Set charge
        # atoms.info.update({"charge": charge, "spin": 1})

        # Add calculator
        atoms.calc = self._predictor

        # Calculate
        energy_kcal = float(atoms.get_potential_energy() * self._ev_to_kcal)
        payload: Dict[str, Any] = {"energy": energy_kcal, "forces": None}

        if gradients:
            forces = atoms.get_forces() * self._ev_to_kcal
            payload["forces"] = forces.tolist()

        return payload


################################################################################
#
# AMP worker
#
# Status: Works for gas-phase single-point energies and forces, but fails for capped h-h structures
#           Heuristic which fixes that is to set USE_MIN_H_H_DISTANCE = True and MIN_H_H_DISTANCE = 0.7
#
# Notes: Uses AMP-BMS utilities.Helpers.build_graph for point calculations
# - The AMP repository structure must be preserved in order to find the PARAMETERS_MIN.yaml file relative to the model path
#
################################################################################

class AMPWorker(TorchBackedMLIPWorker):
    USE_MIN_H_H_DISTANCE = False
    MIN_H_H_DISTANCE = 0.7

    PERIODIC_TABLE = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
        "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
        "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Br": 35, "I": 53
    }

    def load(self) -> None:
        import ase.units
        import torch
        import os

        from utilities.Helpers import load_parameters
        from utilities.Helpers import build_graph

        self._runtime_device = self.setup_torch_runtime(torch)
        self._torch_device = torch.device(self._runtime_device)
        self._build_graph = build_graph

        # go two levels up to find the default config path relative to the model path # FIXME
        default_config_path = os.path.join(os.path.dirname(os.path.dirname(self.model_path)), "parameters", "PARAMETERS_MIN.yaml")
        self.config = load_parameters(default_config_path)
        self.config["device_name"] = str(self._runtime_device)
        self.config["device"] = self._torch_device
        self._dtype = self.config.get("dtype", torch.get_default_dtype())

        self._predictor = self._get_predictor(self.model_path)

        self.cutoff = float(self.config.get("cutoff", 5.0))
        self.cutoff_esp = float(self.config.get("cutoff_esp", 14.0))
        self.cutoff_qmmm_esp = float(self.config.get("cutoff_qmmm_esp", 500.0))
        self.cutoff_qmmm_pol = float(self.config.get("cutoff_qmmm_pol", 9.0))
        self.node_size = int(self.config.get("node_size", 128))
        self.n_channels = int(self.config.get("n_channels", 32))

        self._kj_to_kcal = ase.units.kJ / ase.units.kcal

    def _get_predictor(self, model_name: str) -> Any:
        from amp.AMP import AMP

        predictor = AMP(config=self.config)
        state_dict = self._torch.load(model_name, map_location=self._torch_device, weights_only=False)
        predictor.load_state_dict(state_dict)
        predictor.to(self._torch_device)
        predictor.eval()
        return predictor

    def _parse_xyz(self, xyz: str) -> Tuple[torch.Tensor, torch.Tensor]:
        lines = xyz.strip().splitlines()
        if len(lines) < 2:
            raise ValueError("Invalid XYZ payload")

        try:
            n_atoms = int(lines[0].strip())
        except ValueError as exc:
            raise ValueError("Invalid XYZ atom count") from exc

        if len(lines) < 2 + n_atoms:
            raise ValueError("XYZ payload missing atom coordinates")

        atom_lines = lines[2 : 2 + n_atoms]
        symbols, coords = [], []
        for line_number, line in enumerate(atom_lines, start=3):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid XYZ atom line {line_number}")

            symbol = parts[0].capitalize()
            if symbol not in self.PERIODIC_TABLE:
                raise ValueError(f"Element '{parts[0]}' is not supported by AMP")

            try:
                xyz_coords = [float(value) for value in parts[1:4]]
            except ValueError as exc:
                raise ValueError(f"Invalid coordinate on XYZ atom line {line_number}") from exc

            symbols.append(symbol)
            coords.append(xyz_coords)

        z = self._torch.tensor(
            [self.PERIODIC_TABLE[symbol] for symbol in symbols],
            dtype=self._torch.long,
            device=self._torch_device,
        )
        coords_qm = self._torch.tensor(
            coords,
            dtype=self._dtype,
            device=self._torch_device,
        )
        return z, coords_qm

    def _build_gas_phase_graph(self, z: torch.Tensor, coords_qm: torch.Tensor, charge: int) -> Any:
        coords_qm_batched = coords_qm.unsqueeze(0)
        coords_mm = self._torch.empty((1, 0, 3), dtype=coords_qm.dtype, device=self._torch_device)
        charges_mm = self._torch.empty((1, 0), dtype=coords_qm.dtype, device=self._torch_device)
        graph = self._build_graph(
            Z=z,
            coords_qm=coords_qm_batched,
            coords_mm=coords_mm,
            charges_mm=charges_mm,
            mol_charge=charge,
            cutoff=self.cutoff,
            cutoff_esp=self.cutoff_esp,
            cutoff_qmmm_esp=self.cutoff_qmmm_esp,
            cutoff_qmmm_pol=self.cutoff_qmmm_pol,
            n_channels=self.n_channels,
        )

        if self.USE_MIN_H_H_DISTANCE:
            # Some fragment datasets place link hydrogens unusually close to
            # one another. Optionally exclude those pairs from AMP's singular
            # local Bessel messages while retaining the original long-range
            # electrostatic, dispersion, and repulsion interactions.
            h_h_edge = (z[graph.senders] == 1) & (z[graph.receivers] == 1)
            keep_edge = ~(h_h_edge & (graph.R1[:, 0] < self.MIN_H_H_DISTANCE))
            for attribute in ("R1", "R2", "Rx1", "Rx2", "senders", "receivers"):
                setattr(graph, attribute, getattr(graph, attribute)[keep_edge])

        return graph

    def calculate(self, xyz: str, gradients: bool, charge: int) -> Dict[str, Any]:
        mol_charge = 0 if charge is None else int(charge)
        z, coords_qm = self._parse_xyz(xyz)

        if gradients:
            coords_qm.requires_grad_(True)

        graph = self._build_gas_phase_graph(z=z, coords_qm=coords_qm, charge=mol_charge)

        if gradients:
            with self._torch.enable_grad():
                out = self._predictor(graph)
                force_tensor = -self._torch.autograd.grad(
                    out.V_total,
                    coords_qm,
                    grad_outputs=self._torch.ones_like(out.V_total),
                    allow_unused=True,
                )[0]
                if force_tensor is None:
                    force_tensor = self._torch.zeros_like(coords_qm)
        else:
            with self._torch.no_grad():
                out = self._predictor(graph)
            force_tensor = None

        energy_kcal = float(out.V_total.detach().cpu().reshape(-1)[0]) * self._kj_to_kcal
        payload: Dict[str, Any] = {"energy": energy_kcal, "forces": None}

        if force_tensor is not None:
            forces = force_tensor.detach().cpu().reshape(-1, 3) * self._kj_to_kcal
            payload["forces"] = forces.tolist()

        return payload
