import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Paths
os.makedirs("model", exist_ok=True)
DATA_PATH = "data.csv"
MODEL_PATH = "model/travel_model.pt"
VECTORIZER_PATH = "model/vectorizer.pkl"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"

# Load and clean data
df = pd.read_csv(DATA_PATH)
df.dropna(subset=["Description", "Name"], inplace=True)

# Inputs = Descriptions, Labels = Names
X_text = df["Description"]
y_label = df["Name"]

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text).toarray()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_label)

# Save vectorizer & encoder
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(label_encoder, LABEL_ENCODER_PATH)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# Define model
class TravelModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Train model
model = TravelModel(X_train.shape[1], len(label_encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 1000
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print("✅ Model saved to", MODEL_PATH)
