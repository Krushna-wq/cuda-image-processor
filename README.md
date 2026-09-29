# CUDA Batch Image Grayscale Processor

A GPU-accelerated application for converting batches of RGB PPM images into grayscale PGM images using custom C++/CUDA kernels.

## Objective
To demonstrate an efficient CUDA pipeline that handles CPU-to-GPU memory transfer, runs concurrent pixel conversion through a 2D grid/block thread arrangement, and saves results reliably.

## Technologies Used
- C++17
- NVIDIA CUDA (v12.x+)
- Python 3 (solely for sample dataset generation)
- ## Execution Results

The project was executed successfully on an NVIDIA Tesla T4 GPU in Google Colab.

- Input images: 100
- Output images: 100
- Resolution: 512 x 512
- Total pixels processed: 26,214,400
- CUDA block size: 16 x 16
- CUDA grid size: 32 x 32
- GPU: NVIDIA Tesla T4
- CUDA version: 12.8
- GPU kernel execution time: ~1.80 ms
- Total processing time: ~258.35 ms

All 100 input RGB PPM images were successfully converted into 100 grayscale PGM images.

## CPU -> GPU -> CPU Data Flow
1. **CPU**: Reads PPM image from storage, extracts dimensions, prepares raw RGB 1D byte vector.
2. **GPU (Host -> Device)**: Allocates device pointers via `cudaMalloc`, copies source pixel array using `cudaMemcpy` (Host to Device).
3. **GPU (Execution)**: Launches `rgbToGrayscaleKernel` with grid size dynamically calculated from dimensions and 16x16 CUDA blocks.
4. **GPU (Device -> Host)**: Copies result byte array back to host memory using `cudaMemcpy` (Device to Host).
5. **CPU**: Writes raw byte stream into output directory as PGM grayscale format.

## Kernel & Algorithm
The custom CUDA kernel maps each pixel index in a 2D grid:
```cpp
__global__ void rgbToGrayscaleKernel(const unsigned char* d_rgb, unsigned char* d_gray, int width, int height);
```
It applies the ITU-R BT.601 standard luminance calculation:
$$\text{Gray} = 0.299 \times R + 0.587 \times G + 0.114 \times B$$

## How to Build & Run
1. **Compile & Run using helper script**:
   ```bash
   ./run.sh
   ```
2. **Or compile manually**:
   ```bash
   make
   ./image_processor --input input --output output --images 100
   ```

## Lessons Learned
- **Coalesced Memory Access**: Storing raw RGB sequential data lets GPU warps read contiguous blocks from global memory efficiently.
- **Error Handlers**: Guarding memory operations with `cudaError_t` checks isolates allocation/boundary faults early.
