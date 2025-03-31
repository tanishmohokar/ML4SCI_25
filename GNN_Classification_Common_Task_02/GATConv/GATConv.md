# GATConv

### Model Architecture

    GNN(

            (conv1): GATConv(in_features, hidden_dim)

            (conv2): GATConv(hidden_dim, 2 \* hidden_dim)

            (conv3): GATConv(2 \* hidden_dim, hidden_dim)

            (pool): GlobalMeanPool

            (fc1): Linear(in_features=hidden_dim, out_features=hidden_dim // 4,
            bias=True)

            (activation1): ReLU

            (fc2): Linear(in_features=hidden_dim // 4,out_features=num_classes,
            bias=True)

            (loss_fn): CrossEntropyLoss

      )

### Results

Saved Best Model: Epoch 23, Val. Acc.: 0.7075

Epoch 0, Train Loss: 38.5419, Train Acc: 0.5967, Test Acc: 0.5945, ROC AUC: 0.7215
