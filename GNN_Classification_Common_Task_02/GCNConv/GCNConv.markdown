# GCNConv

### Model Architecture

      GNN(

            (conv1): GCNConv(in_features, hidden_dim)

            (conv2): GCNConv(hidden_dim, 2 \* hidden_dim)

            (conv3): GCNConv(2 \* hidden_dim, hidden_dim)

            (pool): GlobalMeanPool

            (fc1): Linear(in_features=hidden_dim, out_features=hidden_dim // 4,
            bias=True)

            (activation1): ReLU

            (fc2): Linear(in_features=hidden_dim // 4,out_features=num_classes,
            bias=True)

            (loss_fn): CrossEntropyLoss

      )

### Results

Saved Best Model: Epoch 23, Val. Acc.: 0.7185

Epoch 0, Train Loss: 38.9843, Train Acc: 0.6038, Test Acc: 0.5970, ROC
AUC: 0.7097
