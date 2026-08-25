# Diffusion Models From Scratch: DDPM → Latent Diffusion

Two PyTorch notebooks building up the components behind Stable Diffusion, from a plain
pixel-space DDPM to an unconditional Latent Diffusion Model (LDM). Everything is
implemented from scratch (no `diffusers`) — UNet, noise scheduler, VAE, training loops,
and sampling.

| Notebook | What it builds |
|---|---|
| [`Latent_Diffusion_Model_PyTorch_Low_Compute_.ipynb`](./Latent_Diffusion_Model_PyTorch_Low_Compute_.ipynb) | Same Latent Diffusion model but with low processing parameters to "run" with my broke build. |
| [`Latent_Diffusion_Model_PyTorch.ipynb`](./Latent_Diffusion_Model_PyTorch.ipynb) | An **unconditional Latent Diffusion Model** — a KL-regularized VAE compresses images into a small latent grid, and the same diffusion machinery runs *inside that latent space* instead of on raw pixels. This is the architecture Stable Diffusion is built on, minus text conditioning. |

## Why latent diffusion?

Running diffusion directly on pixels means the UNet has to process every pixel at every
one of hundreds of denoising steps — expensive, and most of that signal is
high-frequency detail that barely affects perceptual quality. Latent diffusion instead:

1. Trains a **VAE** to compress images into a much smaller latent grid (here, 4x smaller
   per side, i.e. 16x fewer spatial locations).
2. Trains the **diffusion UNet entirely in that latent space**.
3. At sampling time, denoises pure noise into a latent, then decodes it back to pixels
   with the VAE decoder.

```
Image (H x W x 3) --[VAE Encoder]--> Latent (h x w x c) --[+noise, T steps]--> Noisy latent
Noisy latent --[UNet, T reverse steps]--> Clean latent --[VAE Decoder]--> Image (H x W x 3)
```

Because the UNet never touches full-resolution pixels, this is what lets diffusion
models scale to large (512x512+) images without exploding compute.

## Contents

### `DDPM_PyTorch.ipynb`
- Linear beta noise scheduler (`add_noise`, `sample_prev_timestep`)
- Sinusoidal timestep embeddings
- A small UNet (down/bottleneck/up, with skip connections) predicting noise
- Standard DDPM training loop (predict-the-noise, MSE loss)
- Ancestral sampling loop (T → 0) with visualization

### `Latent_Diffusion_Model_PyTorch.ipynb`
- **Stage 1 — Autoencoder**: `Encoder`/`Decoder` with residual blocks, GroupNorm + SiLU,
  strided down/upsampling; `VAE` wraps them with the reparameterization trick; trained
  with L1 reconstruction loss + a small-weight KL term
- Latent scaling factor computation (normalizes latent variance before diffusion training,
  same trick used in Stable Diffusion)
- **Stage 2 — Latent UNet**: same scheduler/training loop as the DDPM notebook, but
  operating on latents, with self-attention blocks added at the smaller latent
  resolutions (affordable there, too expensive at full pixel resolution)
- Sampling: reverse diffusion in latent space → VAE decode → image
- `FAST_MODE` config flag for a quick (~single-digit-minutes) end-to-end sanity run
  before committing to a full training run

## Requirements

```bash
pip install torch torchvision datasets matplotlib tqdm
```

Both notebooks default to the `nielsr/CelebA-faces` dataset via 🤗 `datasets`, with an
MNIST option as well (`DATASET = "mnist"` in the config cell). Designed to run on a
Colab T4, but any CUDA GPU (or CPU, slowly) works.

## Usage

1. Open either notebook in Colab or Jupyter.
2. Set `DATASET` (`"mnist"` or `"faces"`) in the config cell.
3. In the LDM notebook, `FAST_MODE = True` runs a quick, low-quality end-to-end pass
   to confirm the pipeline works. Set `FAST_MODE = False` for a proper training run.
4. Run all cells top to bottom — VAE trains first (LDM notebook only), then the
   diffusion UNet, then sampling.


## Acknowledgements

Architecture references: Ho et al., *Denoising Diffusion Probabilistic Models* (2020);
Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (2022).
