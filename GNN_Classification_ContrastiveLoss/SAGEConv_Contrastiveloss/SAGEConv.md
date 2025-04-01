# SAGEConv

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
        (conv1): SAGEConv(in_features=hidden_dim, out_features=hidden_dim)  
        (conv2): SAGEConv(in_features=hidden_dim, out_features=hidden_dim)  
        (conv3): SAGEConv(in_features=hidden_dim, out_features=out_dim)  
        (pool): GlobalMeanPool  
    )


### Results

Saved Best Model: Epoch 37, Val. Acc.: 0.7065

Final Test Accuracy: 0.6870

Epoch 1/50 | Loss: 2.7709 | Train Acc: 0.7644 | Val Acc: 0.6235 | Val ROC-AUC: 0.6501
