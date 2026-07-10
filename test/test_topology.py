"""Pure-Python topology tests for non-divisible chiplet partitions."""
import unittest

from core.nop import Nop
from core.topology import (
    balanced_partition_sizes,
    partition_boundaries,
    partition_index,
)


class TopologyTests(unittest.TestCase):
    def test_balanced_partition_sizes_are_left_heavy(self):
        sizes = balanced_partition_sizes(5, 2)

        self.assertEqual(sizes, (3, 2))
        self.assertEqual(partition_boundaries(sizes), (3,))
        self.assertEqual([partition_index(idx, sizes) for idx in range(5)], [0, 0, 0, 1, 1])

    def test_invalid_partitions_fail(self):
        invalid_cases = [(0, 1), (4, 0), (4, 5)]

        for total, partitions in invalid_cases:
            with self.subTest(total=total, partitions=partitions):
                with self.assertRaises(ValueError):
                    balanced_partition_sizes(total, partitions)

    def test_nop_hops_use_partition_boundaries(self):
        nop = Nop(
            nop_bw=1,
            noc_bw=1,
            dram_bw=1,
            dram_list=[(0, 0)],
            nop_hop_cost=1,
            noc_hop_cost=1,
            DRAM_acc_cost=1,
            shape=(5, 1),
            xcut=2,
            ycut=1,
        )

        self.assertEqual(nop.NoP_link_calc((2, 0), (3, 0)), 0)
        self.assertEqual(nop.NoP_link_calc((3, 0), (4, 0)), 1)
        self.assertEqual(nop.NoP_link_calc((1, 0), (5, 0)), 1)
        self.assertEqual(nop.NoP_link_calc((0, 0), (5, 0)), 2)
        self.assertEqual(nop.NoP_link_calc((0, 0), (6, 0)), 3)

    def test_link_indices_include_dram_columns_in_row_stride(self):
        nop = Nop(
            nop_bw=1,
            noc_bw=1,
            dram_bw=1,
            dram_list=[(0, 0)],
            nop_hop_cost=1,
            noc_hop_cost=1,
            DRAM_acc_cost=1,
            shape=(2, 2),
            xcut=1,
            ycut=1,
        )

        row0_right_dram = nop.get_link_idx(3, 0, nop.RIGHT)
        row1_first_core = nop.get_link_idx(1, 1, nop.RIGHT)
        self.assertNotEqual(row0_right_dram, row1_first_core)


if __name__ == '__main__':
    unittest.main()
