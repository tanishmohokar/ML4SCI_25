# common Task 01: Variational Autoencoder (VAE) for Jet Image Data

### Task:To reconstruct a jet image using Autoencoder.

![Model Diagram](https://github.com/tanishmohokar/ML4SCI_25/raw/main/Autoencoder_Common_Task_01/pipeline2.jpg)

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

      VAE_autoencoder(

         (encoder): Sequential(
              (conv1): Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
              (batchnorm1): BatchNorm2d(32)
              (activation1): LeakyReLU(0.2)
        
              (conv2): Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
              (batchnorm2): BatchNorm2d(64)
              (activation2): LeakyReLU(0.2)

              (conv3): Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
              (batchnorm3): BatchNorm2d(128)
              (activation3): LeakyReLU(0.2)

              (conv4): Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
              (batchnorm4): BatchNorm2d(256)
              (activation4): LeakyReLU(0.2)
         )

         (mean_layer): Linear(16384, embedding_dim)
         (logvar_layer): Linear(16384, embedding_dim)
         (reparam): Reparameterization

         (decoder): Sequential(
               (fc1): Linear(embedding_dim, 16384)

               (deconv1): ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
               (batchnorm5): BatchNorm2d(128)
               (activation5): LeakyReLU(0.2)

               (deconv2): ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
               (batchnorm6): BatchNorm2d(64)
               (activation6): LeakyReLU(0.2)

               (deconv3): ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
               (batchnorm7): BatchNorm2d(32)
               (activation7): LeakyReLU(0.2)

               (deconv4): ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
               (activation8): Tanh
         )

         (loss_fn): Variational Loss (KL Divergence + Reconstruction Loss)

      )


- **🎯 Key Insights**

The VAE reconstructs input images by learning a probabilistic latent representation. Evaluation metrics include:
  - **Reconstruction Loss**: Measures the difference between input and output.  
  - **Latent Space Analysis**: Investigates the learned distributions in the bottleneck layer.
  - **KL Divergence Loss**: Regularizes the latent space to match a Gaussian distribution.




