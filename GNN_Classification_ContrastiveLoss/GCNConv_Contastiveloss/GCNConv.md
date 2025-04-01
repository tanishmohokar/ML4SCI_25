# GCNConv

### Model Architecture

    (GraphEncoder)( 
        (point_encoder): Sequential(  
            (0): Linear(in_features=3, out_features=hidden_dim // 2, bias=True)  
            (1): ReLU  
           )  
        (metadata_encoder): Sequential(  
            (0): Linear(in_features=2, out_features=hidden_dim // 4, bias=True)  
            (1): ReLU  
            (2): Linear(in_features=hidden_dim // 4, out_features=hidden_dim // 2, bias=True)  
            )  
        (conv1): GCNConv(in_features=hidden_dim, out_features=hidden_dim)  
        (conv2): GCNConv(in_features=hidden_dim, out_features=hidden_dim)  
        (conv3): GCNConv(in_features=hidden_dim, out_features=out_dim)  
        (pool): GlobalMeanPool  
    )


### Results

Saved Best Model: Epoch 28, Val. Acc.: 0.7100

Final Test Accuracy: 0.6900

Epoch 1/50 | Loss: 2.7707 | Train Acc: 0.7614 | Val Acc: 0.6650 | Val ROC-AUC: 0.7020
