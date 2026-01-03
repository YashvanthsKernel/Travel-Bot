import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib

# Load data and assets
df = pd.read_csv("data.csv")
vectorizer = joblib.load("model/vectorizer.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

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

# Load model
input_dim = len(vectorizer.get_feature_names_out())
output_dim = len(label_encoder.classes_)
model = TravelModel(input_dim, output_dim)
model.load_state_dict(torch.load("model/travel_model.pt"))
model.eval()

# Streamlit UI
st.set_page_config(page_title="ML TravelBot", page_icon="🧠")
st.title("🧠 AI TravelBot (ML Powered)")
st.write("Ask a travel question and get recommendations!")

query = st.text_input("💬 Enter your travel request:")

if query:
    X = vectorizer.transform([query]).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        output = model(X_tensor)
        pred = torch.argmax(output, dim=1)
        place_name = label_encoder.inverse_transform(pred.numpy())[0]

    # Find the full info
    result = df[df["Name"] == place_name].iloc[0]

    st.subheader("🤖 TravelBot Recommendation")
    st.markdown(f"""
    - **Name**: {result['Name']}
    - **Category**: {result['Category']} ({result['Sub_Category']})
    - **Location**: {result['Location_Area']}
    - **Price Range**: {result['Price_Range_INR']}
    - **Rating**: {result['Rating']}
    - **Description**: {result['Description']}
    - **Timing**: {result['Contact_Timings']}
    """)
