"""A collection of MLIP worker implementations for different backends."""

# GUIDE:
# - Each worker class should inherit from MLIPWorker and implement the load() and calculate() methods.
# - The load() method should handle loading the model and preparing it for inference.
# - The calculate() method should take an XYZ string, a gradients flag, and an optional charge, and return a dictionary with "energy" and optionally "forces".
# - Use the resolve_torch_device and apply_torch_limits static methods from MLIPWorker for PyTorch-based workers to handle device selection and resource limits.
# - Ensure that all workers return energy in kcal/mol and forces in kcal/mol/Å for consistency. Use ase.units for unit conversions if needed.
#
# - Add import to mlip_worker_server.py and register the worker class in the WORKER_CLASSES dictionary
# - Add the worker name to keywords.yaml

import abc
import io
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

    @staticmethod # FIXME: Maybe not neccesary because python has already been started
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


################################################################################
#
# TORCHMDNET worker
#
# Status: Works (not tested with gradients and gpu)
#
# Notes: WARNING: Keyword "charge" not found in the input, using default value "0".
#
################################################################################

class TorchMDNetWorker(MLIPWorker):
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

        runtime_device = self.resolve_torch_device(torch_module=torch, requested=self.device)
        self.apply_torch_limits(
            torch_module=torch,
            cpu_threads=self.cpu_threads,
            cuda_memory_fraction=self.cuda_memory_fraction,
            runtime_device=runtime_device,
        )

        self._torch = torch
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
# Status: Fails
#
# Notes:
#
################################################################################

class AimnetWorker(MLIPWorker):
    def load(self) -> None:
        import ase.units
        import torch
        from aimnet.calculators import AIMNet2Calculator

        # torch_device = self.resolve_torch_device(torch_module=torch, requested=self.device)
        # self.apply_torch_limits(
        #     torch_module=torch,
        #     cpu_threads=self.cpu_threads,
        #     cuda_memory_fraction=self.cuda_memory_fraction,
        #     runtime_device=torch_device,
        # )
        # runtime_device = "cuda" if torch_device.startswith("cuda") else "cpu"

        self._dsf = 0
        #self._runtime_device = runtime_device
        self._predictor = AIMNet2Calculator(self.model_path)
        self._ev_to_kcal = ase.units.mol / ase.units.kcal

    def _pybel2data(self, mol, charge):
        import numpy as np

        coord = np.array([a.coords for a in mol.atoms])
        numbers = np.array([a.atomicnum for a in mol.atoms])
        data = {}
        data['coord'] =  coord
        data['numbers'] = numbers
        data['charge'] = charge
        return data

    def calculate(self, xyz: str, gradients: bool, charge: int) -> Dict[str, Any]:
        from openbabel import pybel
        
        # Read molecule
        geostr = xyz.rstrip() + '\n'
        mol = pybel.readstring('xyz', geostr)

        # Set charge
        data = self._pybel2data(mol, charge)

        # Add calculator
        calc = self._predictor

        # Damped shifted DSF electrostatics
        if self._dsf == 1:
            calc.set_lrcoulomb_method('dsf', cutoff=15.0, dsf_alpha=0.2)

        # Calculate
        results = calc(data, forces=False, stress=False, hessian=False)
        energy_kcal = float(results['energy'] * self._ev_to_kcal)
        payload: Dict[str, Any] = {"energy": energy_kcal, "forces": None}

        if gradients:
            forces = results['forces'] * self._ev_to_kcal
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
# Notes: WARNING:root:If 'dataset_list' is provided in the config, the code
#   assumes that each dataset maps to itself. Please use 'dataset_mapping' as
#  'dataset_list' is deprecated and will be removed in the future.
#
################################################################################

class FairchemWorker(MLIPWorker):
    def load(self) -> None:
        import ase.units
        import torch
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        torch_device = self.resolve_torch_device(torch_module=torch, requested=self.device)
        self.apply_torch_limits(
            torch_module=torch,
            cpu_threads=self.cpu_threads,
            cuda_memory_fraction=self.cuda_memory_fraction,
            runtime_device=torch_device,
        )
        runtime_device = "cuda" if torch_device.startswith("cuda") else "cpu"

        self._runtime_device = runtime_device
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
# Status: Fails (not tested with gradients and gpu)
#
# Notes:
# - FIXME: Worker broken on devices with low number of CPU cores - my guess is that MACE codebase has bugs
# - TODO: Add precision flag / or kwargs for all worker types to handle this in a more generic way
# - TODO: Add support for -off and -anicc MACE models
#
################################################################################

class MACEWorker(MLIPWorker):
    def load(self) -> None:
        import ase.units
        import torch
        import mace
        from mace.calculators import MACECalculator, mace_polar, mace_off, mace_anicc

        torch_device = self.resolve_torch_device(torch_module=torch, requested=self.device)
        self.apply_torch_limits(
            torch_module=torch,
            cpu_threads=self.cpu_threads,
            cuda_memory_fraction=self.cuda_memory_fraction,
            runtime_device=torch_device,
        )
        runtime_device = "cuda" if torch_device.startswith("cuda") else "cpu"

        self._runtime_device = runtime_device
        self._predictor = mace_polar(model=self.model_path, device=self._runtime_device) #, return_raw_model=True, default_dtype=self.precision)
        self._ev_to_kcal = ase.units.mol / ase.units.kcal
        print(f"Loaded MACE model on {self._runtime_device} device")

    def calculate(self, xyz: str, gradients: bool, charge: int) -> Dict[str, Any]:
        import ase.io

        # Read molecule
        atoms = ase.io.read(io.StringIO(xyz), format="xyz", index=0)

        # Set charge
        atoms.info.update({"charge": charge, "spin": 1}) #, "external_field": [0.0, 0.0, 0.0]})
        
        # Add calculator
        atoms.calc = self._predictor
        print("Calculating energy and forces with MACE model...")

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
# - TODO: Add torch device handling and torch limits like in fairchem worker
# - FIXME: Doesn't work with zmq
#
# If you have several graphs, batch them like so:
# graph = atoms_adapter.batch([graph1, graph2])
# or 
# graph = atoms_adapter.from_ase_atoms_list([atoms1, atoms2])
#
################################################################################

class OrbitalWorker(MLIPWorker):
    def load(self) -> None:
        import ase.units
        from orb_models.forcefield.inference.calculator import ORBCalculator
        from orb_models.common.utils import seed_everything

        self._predictor, self._atoms_adapter = self._get_predictor(self.model_path, precision="float32-high") # or "float32-highest" / "float64 
        seed_everything(42)
        self._calculator_cls = ORBCalculator
        self._ev_to_kcal = ase.units.mol / ase.units.kcal

    def _get_predictor(self, model_name: str, precision: str) -> Any:
        from orb_models.forcefield import pretrained

        if model_name == "orb-v3-conservative-omol":
            return pretrained.orb_v3_conservative_omol(device=self.device, precision=precision)
        elif model_name == "orb-v3-direct-omol":
            return pretrained.orb_v3_direct_omol(device=self.device, precision=precision)
        else:
            # Fallback for materials if you really intended to use them, but warn the user
            print(f"Warning: Loading generic/material model {model_name}. Charge might be ignored.")
            # Try to load it dynamically if it exists in pretrained
            if hasattr(pretrained, model_name.replace("-", "_")):
                return getattr(pretrained, model_name.replace("-", "_"))(device=self.device, precision=precision)
            else:
                raise ValueError(f"Model {model_name} not found.")

    def calculate(self, xyz: str, gradients: bool, charge: int) -> Dict[str, Any]:
        import ase.io

        # Read molecule
        atoms = ase.io.read(io.StringIO(xyz), format="xyz", index=0)

        # Set charge
        atoms.info.update({"charge": charge, "spin": 1}) #, "external_field": [0.0, 0.0, 0.0]})

        # Add calculator
        atoms.calc = self._calculator_cls(self._predictor, atoms_adapter=self._atoms_adapter, device=self.device)

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
        from so3lr import So3lrCalculator

        self._predictor = self._get_predictor(None, precision=np.float64)
        self._ev_to_kcal = ase.units.mol / ase.units.kcal

    def _get_predictor(self, model_name: str, precision: Any) -> Any:
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

class NequipWorker(MLIPWorker):
    def load(self) -> None:
        import ase.units
        import cuequivariance_torch
        from nequip.integrations.ase import NequIPCalculator

        self._calculator_cls = NequIPCalculator
        self._predictor = self._get_predictor(self.model_path)
        self._kj_to_kcal = ase.units.kJ / ase.units.kcal
        self._ev_to_kcal = ase.units.mol / ase.units.kcal

    def _get_predictor(self, model_name: str) -> Any:
        return self._calculator_cls.from_compiled_model(
            compile_path=model_name,
            device=self.device,
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
            forces = atoms.get_forces() * self._ev_to_kcal
            payload["forces"] = forces.tolist()

        return payload