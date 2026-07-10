"""Pure-Python golden test for Statistic's structured energy boundary."""
import unittest

import numpy as np

from core.statistic import Statistic


class StatisticSnapshotTests(unittest.TestCase):
    def test_snapshot_matches_legacy_cost_and_inclusive_total(self):
        monitor = Statistic((1, 1), (1, 1), (4, 4), (4, 4), 1.8e9)
        name = monitor.init_exe_info('toy', 2)
        monitor.latency_dict[name][:] = [10, 20]
        monitor.nop_dict[name][:] = [1, 2]
        monitor.noc_dict[name][:] = [3, 4]
        monitor.dram_dict[name][:] = [5, 6]
        monitor.core_dict[name][0, 0, :] = [1, 2, 3, 4, 5, 6, 7]
        monitor.core_dict[name][1, 0, :] = [10, 20, 30, 40, 50, 60, 70]

        snapshot = monitor.get_nn_breakdown(name)
        latency, legacy_energy = monitor.get_nn_cost(name)

        self.assertEqual(snapshot['latency_cycles'], 30.0)
        self.assertEqual(snapshot['energy_pj'], {
            'nop': 3.0,
            'noc': 7.0,
            'dram': 11.0,
            'compute': 33.0,
            'ubuf': 33.0,
            'input_buffers': 99.0,
            'output_buffers': 143.0,
        })
        self.assertEqual(snapshot['total_energy_pj_including_compute'], 329.0)
        self.assertEqual(snapshot['modeled_energy_pj_excluding_compute'], 296.0)
        self.assertEqual((latency, legacy_energy), (30.0, 296.0))


if __name__ == '__main__':
    unittest.main()
