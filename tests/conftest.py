"""The RCC login node allows 200 threads per user and 6 GB of address space, which zarr's per-core
thread pool exhausts once other sessions run. zarr's pytest plugin imports it before this file."""

import zarr

zarr.config.set({"threading.max_workers": 1})
