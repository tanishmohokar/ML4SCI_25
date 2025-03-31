\# GCNConv

\### Model Architecture

GNN(

(conv1): GCNConv(in\_features, hidden\_dim)

(conv2): GCNConv(hidden\_dim, 2 \* hidden\_dim)

(conv3): GCNConv(2 \* hidden\_dim, hidden\_dim)

(pool): GlobalMeanPool

(fc1): Linear(in\_features=hidden\_dim, out\_features=hidden\_dim // 4, bias=True)

(activation1): ReLU

(fc2): Linear(in\_features=hidden\_dim // 4,out\_features=num\_classes, bias=True)

(loss\_fn): CrossEntropyLoss

)

\### Results

Saved Best Model: Epoch 23, Val. Acc.: 0.7185

Epoch 0, Train Loss: 38.9843, Train Acc: 0.6038, Test Acc: 0.5970, ROC AUC: 0.7097