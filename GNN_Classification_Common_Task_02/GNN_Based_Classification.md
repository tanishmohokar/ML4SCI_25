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
