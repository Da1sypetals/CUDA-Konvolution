# CUDA implementation of Konvolution

- Replacing linear operation in convolution with KAN-like operation

# Warning

- This implementation sometimes reports CUDA illegal memory accessing (my setting is a 8-gpu server where the code runs on `cuda:0` and fails on `cuda:7`).

## References:

- [pytorch implementation](https://github.com/1ssb/torchkan)

## Note

- There are no optimizations in this implementation. I a cuda beginner and willing to receive optimization suggestions : )

## Start

1. Install

```bash
pip install -e .
```

> Make sure the version of nvcc in PATH is compatible with your current PyTorch version (it seems minor version difference is OK).

2. Run

   - Run test on MNIST:

   ```bash
   python test_kal.py
   ```
