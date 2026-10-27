# Non-Uniform Fast Fourier Transform (NUFFT)

Implementation of the **Non-Uniform Fast Fourier Transform (NUFFT)** based on the approach of Dutt and Rokhlin (1993).

### Project Overview

Implemented NUFFT algorithms for handling Fourier transforms of data sampled at **non-uniform points**, including:

- **Type I NUFFT** – non-uniform spatial points → uniform Fourier modes
- **Type II NUFFT** – uniform Fourier modes → non-uniform spatial points
- **Type III NUFFT** – non-uniform spatial points → non-uniform frequency points
- Direct computation for comparison and validation
- FFT-based acceleration using a self-implemented FFT
- Error analysis and comparison with direct computation and `finufft`

### Implementation

The implementation uses Gaussian-based spreading/interpolation and FFT-based convolution to efficiently approximate the non-uniform Fourier transform.

The project includes experiments with different parameter choices, including the kernel width and truncation parameters, and evaluates the accuracy of the resulting approximation.

### Technologies

**Python · NumPy · FFT · NUFFT · Numerical Methods**

### References

Dutt, A. & Rokhlin, V. (1993). *Fast Fourier Transforms for Nonequispaced Data.*
