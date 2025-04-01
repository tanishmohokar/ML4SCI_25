# GATConv

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
        (conv1): GATConv(in_features=hidden_dim, out_features=hidden_dim)  
        (conv2): GATConv(in_features=hidden_dim, out_features=hidden_dim)  
        (conv3): GATConv(in_features=hidden_dim, out_features=out_dim)  
        (pool): GlobalMeanPool  
    )


### Results

Saved Best Model: Epoch 43, Val. Acc.: 0.7025

Final Test Accuracy: 0.6925

Epoch 1/50 | Loss: 2.7707 | Train Acc: 0.7521 | Val Acc: 0.6170 | Val ROC-AUC: 0.6599
