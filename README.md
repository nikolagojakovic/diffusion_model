# Diffusion Model Training on CIFAR-10 with PyTorch

## Project Overview

This project demonstrates how to implement and train a **Denoising Diffusion Probabilistic Model (DDPM)** on the CIFAR-10 dataset using PyTorch. Diffusion models are generative models that learn to gradually denoise data, generating realistic images from pure noise.

---

## Key Features

- **Data preprocessing:**  
  CIFAR-10 images are resized to 64×64 pixels and normalized to the range \([-1, 1]\) to help stabilize training.

- **Forward diffusion process:**  
  We add Gaussian noise to clean images over \(T\) discrete timesteps, using a **linear beta schedule** \(\beta_t\) that controls noise variance:

  $$
  \beta_t = \text{linearly spaced between } 0.0001 \text{ and } 0.02, \quad t=1, \ldots, T
  $$

  At timestep \(t\), the noisy image \(x_t\) is sampled as:

  $$
  x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon,
  $$

where:

- \(x_0\) is the original clean image,  
- \(\epsilon \sim \mathcal{N}(0, I)\) is Gaussian noise,  
- \(\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)\) is the cumulative product of \(\alpha_t = 1 - \beta_t\).



- **Batch processing:**  
  The implementation supports efficient batch-wise computation by dynamically indexing noise schedule parameters for each sample's timestep.

- **Visualization:**  
  Utilities convert noisy tensors back to images for easy visualization of noise progression during diffusion.

- **Training pipeline:**  
  A PyTorch `DataLoader` provides mini-batches with data augmentation (random horizontal flips), preparing data for neural network training to learn the reverse diffusion.

---

## Why This Matters

Diffusion models have recently emerged as state-of-the-art generative models, capable of producing high-quality, diverse images. This project implements the forward (noising) process, which is essential for training models to learn the reverse (denoising) process and generate new images.

---

## Usage

- Prepare CIFAR-10 dataset with the provided preprocessing pipeline.  
- Use the forward diffusion functions to generate noisy samples at different timesteps.  
- Integrate with a neural network to train on noise prediction and learn the reverse diffusion.

---

Feel free to customize the noise schedule, image size, or augmentations to suit your experiments!

---

If you'd like, I can help add the reverse diffusion sampling and model training code next.
