"""POLAR per-system controls survive the standard graph conversion path."""

import numpy as np

from mace_jax.data.utils import (
    Configuration,
    get_atomic_number_table_from_zs,
    graph_from_configuration,
)
from mace_jax.tools.gin_model import _graph_to_data


def test_charge_spin_and_field_reach_model_inputs():
    configuration = Configuration(
        atomic_numbers=np.asarray([8, 1]),
        positions=np.asarray([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]),
        cell=np.eye(3) * 10.0,
        pbc=(False, False, False),
        total_charge=1.0,
        total_spin=2.0,
        fermi_level=0.15,
        external_field=np.asarray([0.01, -0.02, 0.03]),
    )
    graph = graph_from_configuration(
        configuration, cutoff=3.0, z_table=get_atomic_number_table_from_zs([1, 8])
    )
    data = _graph_to_data(graph, num_species=2)
    np.testing.assert_array_equal(np.asarray(data['total_charge']), [1.0])
    np.testing.assert_array_equal(np.asarray(data['total_spin']), [2.0])
    np.testing.assert_array_equal(np.asarray(data['fermi_level']), [0.15])
    np.testing.assert_array_equal(
        np.asarray(data['external_field']), [[0.01, -0.02, 0.03]]
    )
    np.testing.assert_array_equal(np.asarray(data['pbc']), [[False, False, False]])
