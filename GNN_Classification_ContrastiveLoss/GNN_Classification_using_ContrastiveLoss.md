# Quark/Gluon Classification using Contrastive Loss

### Task:  
To classify input as Quark/Gluon using Graph Neural Networks (GNN).

### Approach:  
![Graph Formation](https://raw.github.com/tanishmohokar/ML4SCI_25/main/GNN_Classification_ContrastiveLoss/graph_formation.png)

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
![Model Architecture](https://raw.github.com/tanishmohokar/ML4SCI_25/main/GNN_Classification_ContrastiveLoss/Pipeline.png)

| Model Name | Accuracy | Notebook Link | PDF Link | Loss | Readme | ROC AUC Score |
|------------|----------|---------------|----------|------|--------|--------------|
| GCNConv | [70.40%](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GCNConv_Contastiveloss/Accuracy_GCN_Contrastive.png) | [Notebook](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GCNConv_Contastiveloss/GCN_Model_Contrastive.ipynb) | [PDF](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GCNConv_Contastiveloss/GCN_Model_Contrastive.pdf) | [LOSSES](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GCNConv_Contastiveloss/Loss_GCN_Contrastive.png) | [readme](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GCNConv_Contastiveloss/GCNConv.md) | [0.7462](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GCNConv_Contastiveloss/ROC_GCN_Contrastive.png) |
| GATConv | [68.10%](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GATConv_Contrastiveloss/Accuracy_GAT_Contrastive.png) | [Notebook](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GATConv_Contrastiveloss/GAT_Model_Contrastive.ipynb) | [PDF](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GATConv_Contrastiveloss/GAT_Model_Contrastive.pdf) | [LOSSES](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GATConv_Contrastiveloss/Loss_GAT_Contrastive.png) | [readme](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GATConv_Contrastiveloss/GATConv.md) | [0.7336](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GATConv_Contrastiveloss/ROC_GAT_Contrastive.png) |
| SAGEConv | [69.80%](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/SAGEConv_Contrastiveloss/Accuracy_SAGE_Contrastive.png) | [Notebook](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/SAGEConv_Contrastiveloss/SAGE_Model_Contrastive.ipynb) | [PDF](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/SAGEConv_Contrastiveloss/SAGE_Model_Contrastive.pdf) | [LOSSES](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/SAGEConv_Contrastiveloss/Loss_SAGE_Contrastive.png) | [readme](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/SAGEConv_Contrastiveloss/SAGEConv.md) | [0.7326](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/SAGEConv_Contrastiveloss/ROC_SAGE_Contrastive.png) |
| GINConv | [69.95%](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GINConv_Contrastiveloss/Accuracy_GIN_Contrastive.png) | [Notebook](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GINConv_Contrastiveloss/GIN_Model_Contrastive.ipynb) | [PDF](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GINConv_Contrastiveloss/GIN_Model_Contrastive.pdf) | [LOSSES](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GINConv_Contrastiveloss/Loss_GIN_Contrastive.png) | [readme](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GINConv_Contrastiveloss/GINConv.md) | [0.7357](https://github.com/tanishmohokar/ML4SCI_25/blob/main/GNN_Classification_ContrastiveLoss/GINConv_Contrastiveloss/ROC_GIN_Contrastive.png) |
