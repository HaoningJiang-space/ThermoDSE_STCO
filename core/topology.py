"""Shared chiplet partition helpers."""


def balanced_partition_sizes(total, partitions):
    if total <= 0:
        raise ValueError("total must be positive")
    if partitions <= 0:
        raise ValueError("partitions must be positive")
    if partitions > total:
        raise ValueError("partitions cannot exceed total")

    base = total // partitions
    remainder = total % partitions
    return tuple(base + (1 if idx < remainder else 0) for idx in range(partitions))


def partition_boundaries(sizes):
    boundaries = []
    running = 0
    for size in sizes[:-1]:
        running += size
        boundaries.append(running)
    return tuple(boundaries)


def partition_index(core_index, sizes):
    if core_index < 0 or core_index >= sum(sizes):
        raise ValueError("core_index is outside the partitioned range")

    running = 0
    for idx, size in enumerate(sizes):
        running += size
        if core_index < running:
            return idx
    raise ValueError("core_index is outside the partitioned range")
