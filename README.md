# ML4SCI_25
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


# Common Task 02: GNN based Quark-Gluon Classification

### Task:  
To classify input as Quark/Gluon using Graph Neural Networks (GNN).

### Approach:  
![Model Diagram](https://github.com/tanishmohokar/ML4SCI_25/raw/main/GNN_Classification_Common_Task_02/graph_formation.png)

- **🚀 Introduction**  
  - This project explores the classification of quark and gluon jets using **Graph Neural Networks (GNNs)**.  
  - By leveraging advanced deep learning architectures, we transform particle collision data from the **Large Hadron Collider (LHC)** into graph representations, enhancing classification accuracy.  

- **📊 Dataset Overview**  
  - **Stored in an HDF5 file:** `quark-gluon_data-set_n139306.hdf5`  
    - Comprises jet images with **three distinct channels**:  
      - **Particle Tracks**  
      - **Electromagnetic Calorimeter (ECAL) Readings**  
      - **Hadronic Calorimeter (HCAL) Readings**  

- **Preprocessing**  
  - Converting `(3,125,125)` jet image to `(3,128,128)` for better visualization.  
  - Extracts point cloud representations from jet images.  
  - Converts point clouds into graph representations.  
  - Normalizes the dataset and splits it into training and test sets.  

- **⚙️ Model Architectures**  
  - This repository implements multiple **GNN-based models**, including:  
    - **Graph Convolutional Networks (GCN)**  
    - **Graph Attention Networks (GAT)**  
    - **Graph Isomorphism Networks (GIN)**  
    - **GraphSAGE**  

- **🎯 Key Insights**  
  - The models effectively classify quark and gluon jets.  
  - Achieves high accuracy and robustness with different GNN architectures.  

### Model:  
![Model Diagram](https://github.com/tanishmohokar/ML4SCI_25/raw/main/GNN_Classification_Common_Task_02/Pipeline.png)

| Model Name | Accuracy | Notebook Link | PDF Link | Loss | Readme | ROC AUC Score |
|------------|----------|---------------|----------|------|--------|--------------|
| GCNConv | [71.70%](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GCNConv/Accuracy_plot_GCN.png) | [Notebook](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GCNConv/GCNConv-final.ipynb) | [PDF](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GCNConv/GCNConv-final.pdf) | [LOSSES](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GCNConv/Loss_plot_GCN.png) | [readme](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GCNConv/GCNConv.markdown) | [0.7825](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GCNConv/ROC_plot_GCN.png) |
| GATConv | [71.45%](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GATConv/Accuracy_plot_GAT.png) | [Notebook](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GATConv/GATConv-final.ipynb) | [PDF](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GATConv/GATConv-final.pdf) | [LOSSES](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GATConv/Loss_plot_GAT.png) | [readme](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GATConv/GATConv.md) | [0.7835](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GATConv/ROC_plot_GAT.png) |
| SAGEConv | [71.95%](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/SAGEConv/Accuracy_plot_SAGE.png) | [Notebook](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/SAGEConv/SAGEConv-final.ipynb) | [PDF](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/SAGEConv/SAGEConv-final.pdf) | [LOSSES](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/SAGEConv/Loss_plot_SAGE.png) | [readme](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/SAGEConv/SAGEConv.md) | [0.7860](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/SAGEConv/ROC_plot_SAGE.png) |
| GINConv | [71.70%](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GINConv/Accuracy_plot_GIN.png) | [Notebook](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GINConv/GINConv-final.ipynb) | [PDF](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GINConv/GINConv-final.pdf) | [LOSSES](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GINConv/Loss_plot_GIN.png) | [readme](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GINConv/GINConv.md) | [0.7836](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_Common_Task_02/GINConv/ROC_plot_GIN.png) |

# Quark/Gluon Classification using Contrastive Loss

### Task:  
To classify input as Quark/Gluon using Graph Neural Networks (GNN).

### Approach:  
![Graph Formation](https://github.com/tanishmohokar/ML4SCI_25/raw/main/GNN_Classification_ContrastiveLoss/graph_formation.png)

- **🚀 Introduction**  
  - This project contains an implementation of multiple Graph Neural Network (GNN) architectures for classifying quark and gluon jets using contrastive loss.  
  - The models leverage the Large Hadron Collider dataset, transforming jet images into point clouds and then graph representations for classification.  

- **📊 Dataset Overview**  
  - The dataset is stored in an HDF5 file: `quark-gluon_data-set_n139306.hdf5`.  
  - It contains jet images with 3-channel data:  
    - Particle Tracks  
    - Electromagnetic Calorimeter (ECAL) Readings  
    - Hadronic Calorimeter (HCAL) Readings  

- **Preprocessing**  
  - Converting (3,125,125) jet image to (3,128,128) for better visualization  
  - Extracts point cloud representations from jet images  
  - Converts point clouds into graph representations  
  - Normalizes the dataset and splits it into training and test sets  

- **⚙️ Model Architectures**  
  This repository implements multiple GNN-based models, including:  
  - Graph Convolutional Networks (GCN)  
  - Graph Attention Networks (GAT)  
  - Graph Isomorphism Networks (GIN)  
  - GraphSAGE  

- **🎯 Key Insights**  
  - The models effectively classify quark and gluon jets  
  - Uses contrastive loss to improve feature separation  
  - Achieves high accuracy and robustness with different GNN architectures  

### Model:  
![Model Architecture](https://github.com/tanishmohokar/ML4SCI_25/raw/main/GNN_Classification_ContrastiveLoss/Pipeline.png)

| Model Name | Accuracy | Notebook Link | PDF Link | Loss | Readme | ROC AUC Score |
|------------|----------|---------------|----------|------|--------|--------------|
| GCNConv | [71.00%](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GCNConv_Contastiveloss/Accuracy_plot_GCN_Contrastive.png) | [Notebook](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GCNConv_Contastiveloss/GCN-Contrastive-final.ipynb) | [PDF](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GCNConv_Contastiveloss/GCN-Contrastive-final.pdf) | [LOSSES](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GCNConv_Contastiveloss/Loss_plot_GCN_Contrastive.png) | [readme](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GCNConv_Contastiveloss/GCNConv.md) | [0.7302](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GCNConv_Contastiveloss/ROC_plot_GCN_Contrastive.png) |
| GATConv | [70.25%](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GATConv_Contrastiveloss/Accuracy_plot_GAT_Contrastive.png) | [Notebook](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GATConv_Contrastiveloss/GAT-Contrastive-Final.ipynb) | [PDF](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GATConv_Contrastiveloss/GAT-Contrastive-final.pdf) | [LOSSES](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GATConv_Contrastiveloss/Loss_plot_GAT_Contrastive.png) | [readme](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GATConv_Contrastiveloss/GATConv.md) | [0.7369](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GATConv_Contrastiveloss/ROC_plot_GAT_Contrastive.png) |
| SAGEConv | [70.65%](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/SAGEConv_Contrastiveloss/Accuracy_plot_SAGE_Contrastive.png) | [Notebook](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/SAGEConv_Contrastiveloss/SAGE-Contrastive-final.ipynb) | [PDF](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/SAGEConv_Contrastiveloss/SAGE-Contrastive-final.pdf) | [LOSSES](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/SAGEConv_Contrastiveloss/Loss_plot_SAGE_Contrastive.png) | [readme](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/SAGEConv_Contrastiveloss/SAGEConv.md) | [0.7311](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/SAGEConv_Contrastiveloss/ROC_plot_SAGE_Contrastive.png) |
| GINConv | [70.65%](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GINConv_Contrastiveloss/Accuracy_plot_GIN_Contrastive.png) | [Notebook](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GINConv_Contrastiveloss/GIN-Contrastive-final.ipynb) | [PDF](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GINConv_Contrastiveloss/GIN-Contrastive-final.pdf) | [LOSSES](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GINConv_Contrastiveloss/Loss_plot_GIN_Contrastive.png) | [readme](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GINConv_Contrastiveloss/GINConv.md) | [0.7444](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GINConv_Contrastiveloss/ROC_plot_GIN_Contrastive.png) |
