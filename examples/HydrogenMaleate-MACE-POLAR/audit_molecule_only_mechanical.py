"""Compare existing mechanical implementations at identical saved coordinates.

Evaluates forces only; does not integrate or alter either trajectory.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import jax.numpy as jnp
import numpy as np
import openmm
import torch
from ase.io import read
from openmm import app, unit
from openmmml import MLPotential

from md import build_energy
from openmm_mechanical import build_classical_system, build_topology, evaluate, KJ_MOL_PER_EV, TIP3P_Q


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[3]
    example = root/'Examples/HydrogenMaleate-MoleculeOnly'
    initial = read(example/'structure/initial-100ps-full-ml.xyz')
    qm_charges = np.asarray(json.loads((example/'structure/molecule-fixed-charges.json').read_text())['charges'])
    charges = np.concatenate((qm_charges, np.tile(TIP3P_Q,52)))
    topology, _ = build_topology(initial)
    mixed = MLPotential('mace', modelPath=str(root/'artifacts/MACE-POLAR-1-M.model')).createMixedSystem(
        topology, build_classical_system(initial,charges), list(range(11)),
        embedding='mechanical', charge=-1, multiplicity=1, precision='double',
        returnEnergyType='energy', mlLongRange=True, device='cpu', forceGroup=1,
    )
    simulation = app.Simulation(topology, mixed, openmm.VerletIntegrator(.0005),
                                openmm.Platform.getPlatformByName('CPU'), {'Threads':'4'})
    cutoff = min(9., .49*float(initial.cell[0,0]))
    force_eval, params, neighbor_fn, _, _ = build_energy(
        initial, root/'artifacts/MACE-POLAR-1-M-jax-fixed.msgpack',
        'mechanical', .5, 6, 0, 'none', 4.2, .2,
        qm_charges, cutoff, .25, 'pme', .5, 8,
    )
    output = {
        'jax_cutoff_angstrom':cutoff, 'openmm_cutoff_angstrom':5.8,
        'force_rms_definition':'Root mean squared Cartesian component difference',
        'configurations':{},
    }
    saved = np.load(example/'openmm-mechanical-100ps/initial-evaluation.npz')
    output['saved_openmm_initial_position_max_difference_angstrom'] = float(
        np.max(np.abs(saved['positions_angstrom']-initial.positions)))
    frames = {'initial':initial,
              'openmm_final':read(example/'openmm-mechanical-100ps/final.xyz'),
              'jax_final':read(example/'mechanical-100ps/final.xyz')}
    for name, atoms in frames.items():
        start=time.perf_counter()
        r=jnp.asarray(atoms.positions,dtype=jnp.float64)
        energy, grad = force_eval(r,neighbor_fn.allocate(r),params)
        energy.block_until_ready()
        simulation.context.setPositions(atoms.positions*.1*unit.nanometer)
        om_energy, om_force = evaluate(simulation)
        om_energy /= KJ_MOL_PER_EV
        om_force /= KJ_MOL_PER_EV*10
        difference = -np.asarray(grad)-om_force
        distances = atoms.positions[:4]-atoms.positions[10]
        distances -= float(initial.cell[0,0])*np.rint(distances/float(initial.cell[0,0]))
        distances = np.linalg.norm(distances,axis=1)
        output['configurations'][name] = {
            'delta_angstrom':float(distances[2:].min()-distances[:2].min()),
            'jax_energy_eV':float(energy),'openmm_energy_eV':om_energy,
            'energy_difference_meV':1000*(float(energy)-om_energy),
            'force_component_rms_meV_per_angstrom':float(1000*np.sqrt(np.mean(difference**2))),
            'solute_force_component_rms_meV_per_angstrom':float(1000*np.sqrt(np.mean(difference[:11]**2))),
            'max_atom_force_difference_meV_per_angstrom':float(1000*np.linalg.norm(difference,axis=1).max()),
            'evaluation_seconds':time.perf_counter()-start,
        }
        print(name,json.dumps(output['configurations'][name]),flush=True)
    target=root/'Examples/HydrogenMaleate-Analysis/molecule-only-mechanical-force-audit.json'
    target.write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    main()
