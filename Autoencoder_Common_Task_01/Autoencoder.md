# common Task 01: Variational Autoencoder (VAE) for Jet Image Data

### Task:
To reconstruct a jet image using Autoencoder.

- **🚀 Introduction**  
  - This repository contains a PyTorch implementation of a **Variational Autoencoder (VAE)** designed to process jet image data  
  - The model learns a probabilistic latent representation of the input images and reconstructs them using a Gaussian loss function.
  - The goal is to analyze high-energy physics datasets for anomaly detection and feature extraction.

- **Features** 
   - Data Loading: Reads and processes jet image data from an HDF5 file.
   - VAE Architecture: Utilizes convolutional layers for encoding and decoding, with a probabilistic latent space.
   - Gaussian Loss Function: Incorporates a Gaussian loss for improved probabilistic modeling.
   - Dataset Splitting: Includes functionality for training-validation-test data partitioning.
   - Evaluation and Visualization: Generates and displays reconstructed images for qualitative analysis.

- **📊 Dataset Overview**  
  - **Stored in an HDF5 file:** `quark-gluon_data-set_n139306.hdf5`  
    - Comprises jet images with **three distinct channels**:  
      - **Particle Tracks**  
      - **Electromagnetic Calorimeter (ECAL) Readings**  
      - **Hadronic Calorimeter (HCAL) Readings**

- **⚙️ Model Architectures**

- **🎯 Key Insights**
The VAE reconstructs input images by learning a probabilistic latent representation. Evaluation metrics include:
  - **Reconstruction Loss**: Measures the difference between input and output.  
  - **Latent Space Analysis**: Investigates the learned distributions in the bottleneck layer.
  - **KL Divergence Loss**: Regularizes the latent space to match a Gaussian distribution.




