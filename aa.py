import transformers
import torch
import numpy as np
import pandas as pd

def load_files(csv_file="",npy_file=""): return pd.read_csv(csv_file), np.load(npy_file, allow_pickle=True)
print(transformers.__version__)

# load embedding vectors from people_embeddings.npy and load labels from train_connections.csv
train_connections, train_embeddings = load_files(csv_file="train_connections.csv", npy_file="people_embeddings.npy")
labels_list = [1 if connection_true else 0 for connection_true in train_connections['true_attempt']]
labels = np.array(labels_list) 
labels.reshape(-1,1)

train_labels = torch.tensor(labels)
train_embeddings = torch.tensor(train_embeddings)

# load untrained model from transformers not using pretrained
model = transformers.BertForSequenceClassification(config=transformers.BertConfig(), num_labels=1)

# define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.BCELoss()

# train model
model.train()

for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(train_embeddings, labels=train_labels)
    loss = loss_fn(outputs.logits, train_labels)
    loss.backward()
    optimizer.step()
    print(loss.item())

# save model
model.save_pretrained('bert-base-uncased')

# load test embeddings
test_embeddings = np.load("test_embeddings.npy", allow_pickle=True)