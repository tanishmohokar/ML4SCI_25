# GINConv

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
        (conv1): GINConv(Sequential(  
              (0): Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)  
              (1): ReLU  
              (2): Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)  
        ))  
        (conv2): GINConv(Sequential(  
              (0): Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)  
              (1): ReLU  
              (2): Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)  
        ))  
        (conv3): GINConv(Sequential(  
              (0): Linear(in_features=hidden_dim, out_features=out_dim, bias=True)  
              (1): ReLU  
              (2): Linear(in_features=out_dim, out_features=out_dim, bias=True)  
        ))  
        (pool): GlobalMeanPool   
    )


### Results

Saved Best Model: Epoch 29, Val. Acc.: 0.7065

Final Test Accuracy: 0.7065

Epoch 1/50 | Loss: 2.7709 | Train Acc: 0.7642 | Val Acc: 0.6580 | Val ROC-AUC: 0.7186
