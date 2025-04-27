# Distance-Wrap Advanced Sampler for Comfy Diffusion

A custom sampler for the ComfyUI k-diffusion backend that adaptively blends intermediate “derivatives” (denoising steps) using distance-based weights (including softmax or spherical interpolation). Supports Euler, CFG-postprocessing, sharpening/perpendicular steps, smoothing, and negative guidance.

---

## Table of Contents

- [Features](#features)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Quick Start](#quick-start)  
- [API Reference](#api-reference)  
  - [`distance_wrap(…)`](#distance_wrap)  
    - Parameters  
    - Returns  
  - [Internals](#internals)  
    - `fast_distance_weights(…)`  
    - `matrix_batch_slerp(…)`  
- [Example Usage](#example-usage)  
- [License](#license)

---

## Features

- **Adaptive resampling** per noise level  
- **Distance-based weighting** of intermediate derivatives  
- Optional **softmax** weighting or **spherical linear interpolation** (slerp)  
- Supports **CFG-postprocessing** and **negative guidance**  
- Optional **sharpen** and **perpendicular** refinement steps  
- Euler fallback when resampling is disabled or unnecessary  

---

## Requirements

- Python 3.8+  
- [PyTorch](https://pytorch.org/)  
- [ComfyUI k-diffusion backend](https://github.com/comfyanonymous/ComfyUI)  

---

## Installation

```bash
pip install torch comfy-k-diffusion
```

Place the sampler file (e.g. `distance_sampler.py`) in your ComfyUI extensions directory.

---

## Quick Start

```py
from comfy.k_diffusion import sampling
from distance_sampler import distance_wrap

# Create a sampler with custom settings
my_sampler = distance_wrap(
    resample=5,
    resample_end=2,
    cfgpp=True,
    sharpen=True,
    use_softmax=False,
    first_only=False,
    use_slerp=True,
    perp_step=True,
    smooth=False,
    use_negative=False,
)

# Plug into ComfyUI
output = sampling.sample_with_my_sampler(
    model=your_model,
    x=init_noise,
    sigmas=sigma_schedule,
    extra_args={…},
)
```

---

## API Reference

### `distance_wrap(resample, resample_end=-1, cfgpp=False, sharpen=False, use_softmax=False, first_only=False, use_slerp=False, perp_step=False, smooth=False, use_negative=False)`

Returns a `sample_distance_advanced(model, x, sigmas, extra_args=None, callback=None, disable=None)` function that implements the advanced sampling loop.

#### Parameters

| Name             | Type      | Default | Description                                                                                   |
| ---------------- | --------- | ------- | --------------------------------------------------------------------------------------------- |
| `resample`       | `int`     | `-1`    | Base number of inner resampling steps per noise level (–1 = `min(10, N/2)`).                  |
| `resample_end`   | `int`     | `-1`    | Number of resample steps at final sigma (linear interpolation between `resample` & this).    |
| `cfgpp`          | `bool`    | `False` | Enable CFG-postprocessing: stores unconditional denoised sample for guidance.                 |
| `sharpen`        | `bool`    | `False` | Apply a sharpening adjustment between iterations.                                            |
| `use_softmax`    | `bool`    | `False` | Compute weighting via softmax over pairwise distances.                                       |
| `first_only`     | `bool`    | `False` | Only resample on the first inner loop.                                                       |
| `use_slerp`      | `bool`    | `False` | Use spherical interpolation (`matrix_batch_slerp`) instead of linear blending.               |
| `perp_step`      | `bool`    | `False` | Apply a perpendicular diffusion adjustment between steps.                                    |
| `smooth`         | `bool`    | `False` | Apply intensity smoothing (renormalize variance) on intermediate denoised predictions.       |
| `use_negative`   | `bool`    | `False` | Treat `uncond_denoised` as “negative” guidance if provided.                                  |

#### Returns

A function:
```py
def sample_distance_advanced(model, x, sigmas, extra_args=None, callback=None, disable=None) -> Tensor
```
- **`model`**: your diffusion model’s denoiser.  
- **`x`**: current latent tensor.  
- **`sigmas`**: 1D tensor of noise levels.  
- **`extra_args`**: dict with optional `"uncond_denoised"`, `"mask"`, etc.  
- **`callback`**: hook each iteration: `callback({'x', 'i', 'sigma', 'sigma_hat', 'denoised'})`.  
- **`disable`**: pass to `trange(…, disable=disable)` to toggle progress bar.

---

## Internals

### `fast_distance_weights(t: Tensor, use_softmax=False, use_slerp=False, uncond=None)`

1. **Normalize** each derivative matrix by its Frobenius norm.  
2. Compute pairwise **Manhattan** distances in normalized space.  
3. Convert distances to weights:  
   - **Softmax** over scaled distances, or  
   - **Linearly scaled & squared** distances, then normalized.  
4. If `use_slerp=True`, call `matrix_batch_slerp`; else linearly blend:  
   ```py
   blended = (t * weights).sum(dim=0)
   ```
5. Re-scale blended tensor to match original norm.

---

### `matrix_batch_slerp(t: Tensor, tn: Tensor, w: Tensor)`

Performs **batch spherical linear interpolation** (slerp):

- Computes pairwise dot-products between normalized derivatives `tn`.  
- Extracts angles `ω = arccos(dot)`.  
- Blends each slice via  
  \[
    \sum_k rac{\sin(w_k \, ω)}{\sin ω} \; t_k
  \]
- Returns a single blended tensor on the original scale.

---

## Example Usage

```py
# 1. Import
from comfy.k_diffusion.sampling import sample
from distance_sampler import distance_wrap

# 2. Prepare sampler
sampler = distance_wrap(
    resample=8, cfgpp=True, use_slerp=True, sharpen=True
)

# 3. Run sampling
result = sample(
    model=my_model,
    x=initial_noise,
    sigmas=sigma_schedule,
    extra_args={'mask': my_mask},
    sampler_fn=sampler
)

# 4. Decode & save
image = decoder(result)
image.save("output.png")
```

---

## License

MIT License. Feel free to copy, modify, and distribute!
