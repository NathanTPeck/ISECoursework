import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
data = pd.read_csv("Title+Body.csv").fillna("")
text_col = "text"
labels = data["sentiment"].values

# TF-IDF vectorization
tfidf = TfidfVectorizer(ngram_range=(1, 1), max_features=1000)
X = tfidf.fit_transform(data[text_col]).toarray()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define neural network class
class BugReportClassifier(nn.Module):
    def __init__(self, input_size):
        super(BugReportClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

sum_accuracy = 0
sum_precision = 0
sum_recall = 0
sum_f1 = 0
sum_auc = 0

num_iters = 10
for i in range(num_iters):
    # Initialize model
    input_size = X_train.shape[1]
    model = BugReportClassifier(input_size).to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)




    # Training loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            labels = labels.view(-1, 1)  # Reshape for BCE loss
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.view(-1, 1)
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # Convert to NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    sum_accuracy += accuracy_score(y_true, y_pred)
    sum_precision += precision_score(y_true, y_pred)
    sum_recall += recall_score(y_true, y_pred)
    sum_f1 += f1_score(y_true, y_pred)
    sum_auc += roc_auc_score(y_true, y_pred)

accuracy = sum_accuracy / num_iters
precision = sum_precision / num_iters
recall = sum_recall / num_iters
f1 = sum_f1 / num_iters
auc = sum_auc / num_iters

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
