# Distributed and Multi-GPU Examples

The distributed folder demonstrates MPI-aware workflows and multi-GPU decomposition.

## Run

```sh
mpirun -n 2 uv run python -m warp.examples.distributed.example_jacobi_mpi
```

## `example_jacobi_mpi.py`

- MPI-based rank decomposition
- `wp.get_cuda_device()` per rank
- stream and event synchronization
- 2D domain updates with explicit boundary exchanges

Useful when porting small kernels to multi-GPU.

## Related docs

- `docs/user_guide/devices.rst`
- local runtime API around streams/events and peer access
- FAQ section on multi-GPU and MPI

## Notes

- Requires `mpi4py` and CUDA-aware MPI for full behavior.
- Confirm one visible GPU per rank where possible.
- Use pinned buffers for repeated transfer phases when timing.
