# GINConv

### Model Architecture

    GNN(

            (conv1): GINConv(Linear(in_features, hidden_dim) → ReLU → Linear(hidden_dim, hidden_dim))  
            
            (conv2): GINConv(Linear(hidden_dim, 2 * hidden_dim) → ReLU → Linear(2 * hidden_dim, 2 * hidden_dim))  
  
            (conv3): GINConv(Linear(2 * hidden_dim, hidden_dim) → ReLU → Linear(hidden_dim, hidden_dim))  
  
            (pool): GlobalMeanPool  
  
            (fc1): Linear(in_features=hidden_dim, out_features=hidden_dim // 4, bias=True)  
  
            (activation1): ReLU  
  
            (fc2): Linear(in_features=hidden_dim // 4, out_features=num_classes, bias=True)  
          
            (loss_fn): CrossEntropyLoss  

      )

### Results

Saved Best Model: Epoch 35, Val. Acc.: 0.7170

Epoch 0, Train Loss: 40.1402, Train Acc: 0.4953, Test Acc: 0.5015, ROC AUC: 0.6100
