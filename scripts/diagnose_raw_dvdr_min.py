"""Measure the unfloored |div(v)|/3 distribution in a QUOKKA snapshot."""

from __future__ import annotations

import argparse

import numpy as np
import yt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--chunk-z", type=int, default=64)
    args = parser.parse_args()

    ds = yt.load(args.dataset)
    dims = np.asarray(ds.domain_dimensions, dtype=int)
    widths = ds.domain_width.to("cm").value / dims

    raw_min = np.inf
    raw_max = -np.inf
    positive_min = np.inf
    zero_count = 0
    below_1e19 = 0
    below_1e18 = 0
    above_1e12 = 0
    total_count = 0

    for start in range(0, int(dims[2]), args.chunk_z):
        stop = min(start + args.chunk_z, int(dims[2]))
        load_start = max(0, start - 1)
        load_stop = min(int(dims[2]), stop + 1)
        left_edge = ds.domain_left_edge.copy()
        left_edge[2] += load_start * ds.domain_width[2] / dims[2]
        if load_stop == int(dims[2]):
            left_edge[2] = (
                ds.domain_right_edge[2]
                - (load_stop - load_start) * ds.domain_width[2] / dims[2]
            )
        grid = ds.covering_grid(
            level=0,
            left_edge=left_edge,
            dims=(int(dims[0]), int(dims[1]), load_stop - load_start),
        )

        vx = grid[("gas", "velocity_x")].to("cm/s").value
        vy = grid[("gas", "velocity_y")].to("cm/s").value
        vz = grid[("gas", "velocity_z")].to("cm/s").value
        div = (
            np.gradient(vx, widths[0], axis=0)
            + np.gradient(vy, widths[1], axis=1)
            + np.gradient(vz, widths[2], axis=2)
        )
        core = np.abs(div[..., start - load_start : stop - load_start]) / 3.0

        raw_min = min(raw_min, float(np.min(core)))
        raw_max = max(raw_max, float(np.max(core)))
        positive = core[core > 0.0]
        if positive.size:
            positive_min = min(positive_min, float(np.min(positive)))
        zero_count += int(np.count_nonzero(core == 0.0))
        below_1e19 += int(np.count_nonzero(core < 1.0e-19))
        below_1e18 += int(np.count_nonzero(core < 1.0e-18))
        above_1e12 += int(np.count_nonzero(core > 1.0e-12))
        total_count += int(core.size)
        print(f"processed z={start}:{stop}", flush=True)

    print(f"shape={tuple(dims)}")
    print(f"raw_min_s-1={raw_min:.17e}")
    print(f"raw_max_s-1={raw_max:.17e}")
    print(f"positive_min_s-1={positive_min:.17e}")
    print(f"zero_count={zero_count}")
    print(f"below_1e-19={below_1e19}")
    print(f"below_1e-18={below_1e18}")
    print(f"above_1e-12={above_1e12}")
    print(f"total_count={total_count}")


if __name__ == "__main__":
    main()
