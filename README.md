<div align="center">

# 🌍 TravelBot AI

### *Your Intelligent Travel Companion for Exploring India*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Neural_Network-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

*An AI-powered travel recommendation system that understands your travel queries and suggests the perfect destinations, restaurants, hotels, and experiences across India's most iconic cities.*

<br/>

[🚀 Quick Start](#-quick-start) •
[✨ Features](#-features) •
[🏙️ Cities](#️-covered-cities) •
[🧠 How It Works](#-how-it-works) •
[📊 Dataset](#-dataset)

---

</div>

<br/>

## 🎯 Overview

**TravelBot AI** is a machine learning-powered travel assistant that transforms natural language queries into personalized travel recommendations. Simply describe what you're looking for — whether it's *"a peaceful temple in Varanasi"*, *"best biryani in Hyderabad"*, or *"luxury hotels near the beach in Goa"* — and let our neural network find the perfect match for you!

<br/>

<div align="center">

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🗣️  "I want to visit ancient forts with scenic views"          ║
║                          ↓                                       ║
║   🧠  TF-IDF Vectorization + Neural Network Processing           ║
║                          ↓                                       ║
║   📍  Recommendation: Golconda Fort, Hyderabad                   ║
║       ⭐ 4.6/5 • 💰 ₹25-200 • ⏰ 9:00 AM - 5:30 PM                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

</div>

<br/>

## ✨ Features

<table>
<tr>
<td width="50%">

### 🤖 **Intelligent Recommendations**
Natural language understanding powered by TF-IDF vectorization and PyTorch neural networks

### 🏛️ **400+ Curated Places**
Handpicked tourist spots, restaurants, hotels, and shopping destinations

### ⚡ **Real-time Inference**
Instant recommendations through our optimized Streamlit interface

</td>
<td width="50%">

### 🌐 **10 Major Cities**
Comprehensive coverage of India's most popular travel destinations

### 📊 **Rich Information**
Detailed data including ratings, prices, timings, and descriptions

### 🎯 **Category Filtering**
Recommendations across Tourist Spots, Food, Hotels, and Shopping

</td>
</tr>
</table>

<br/>

## 🏙️ Covered Cities

<div align="center">

| City | Highlights | Specialty |
|:----:|:-----------|:----------|
| 🕉️ **Varanasi** | Ghats, Temples, Spiritual Sites | Ganga Aarti, Banarasi Silk |
| 🏖️ **Goa** | Beaches, Churches, Nightlife | Portuguese Heritage, Seafood |
| 🏰 **Udaipur** | Palaces, Lakes, Heritage | Royal Rajasthan Experience |
| 🏛️ **Jaipur** | Forts, Palaces, Markets | Pink City Architecture |
| 💎 **Agra** | Taj Mahal, Mughal Heritage | UNESCO World Heritage Sites |
| ⛵ **Kochi** | Backwaters, Colonial Heritage | Chinese Fishing Nets, Spices |
| 🌊 **Chennai** | Temples, Beaches, Culture | South Indian Traditions |
| 🍚 **Hyderabad** | Forts, Cuisine, Tech Hub | Biryani Capital of India |
| 🏙️ **Delhi** | Monuments, Markets, History | Capital City Grandeur |
| 🎬 **Mumbai** | Bollywood, Marine Drive | Maximum City Vibes |

</div>

<br/>

## 🧠 How It Works

Our recommendation engine uses a **two-stage ML pipeline** to understand your travel queries and find the best matches:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           🧠 ML PIPELINE FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐        │
│   │  🗣️ User     │    │  📝 TF-IDF        │    │  🧠 Neural       │        │
│   │    Query     │ ──▶│   Vectorization   │ ──▶│    Network       │        │
│   └──────────────┘    └───────────────────┘    └──────────────────┘        │
│                                                         │                   │
│                                                         ▼                   │
│   ┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐        │
│   │  📊 Rich     │    │  📍 Place         │    │  🎯 Classification│        │
│   │   Details    │ ◀──│   Recommendation  │ ◀──│    Output        │        │
│   └──────────────┘    └───────────────────┘    └──────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Architecture

```python
class TravelModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),    # Feature extraction
            nn.ReLU(),                     # Non-linearity
            nn.Linear(128, output_dim)     # Classification
        )
```

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Vectorizer** | TF-IDF (sklearn) | Converts text queries to numerical features |
| **Encoder** | LabelEncoder | Maps place names to class indices |
| **Model** | PyTorch Neural Net | Learns query-to-place mappings |
| **Interface** | Streamlit | Interactive web application |

<br/>

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install torch pandas streamlit scikit-learn joblib
```

### Installation

```bash
# Clone the repository
git clone https://github.com/YashvanthsKernel/Travel-Bot.git
cd Travel-Bot/Code

# Train the model (optional - pre-trained model included)
python train_model.py

# Launch the app
streamlit run app.py
```

### Usage

1. 🌐 Open your browser to `http://localhost:8501`
2. 💬 Enter your travel query (e.g., *"romantic sunset views"*)
3. 🎯 Get personalized recommendations with full details!

<br/>

## 📊 Dataset

Our comprehensive dataset covers **407 places** across **10 cities** with detailed information:

<div align="center">

```
📁 Dataset/
├── 📊 10 cities.csv         # Complete merged dataset
├── 🕉️ VARANASI.csv          # 52 places
├── 🏖️ GOA (PANJIM).csv      # 55 places
├── 🏰 UDAIPUR.csv           # 44 places
├── 🏛️ JAIPUR.csv            # 38 places
├── 💎 AGRA.csv              # 86 places (2 datasets merged)
├── ⛵ KOCHI.csv             # 44 places
├── 🌊 CHENNAI.csv           # 44 places
├── 🍚 HYDERABAD.csv         # 54 places
└── 📅 SEASONS.csv           # Best travel seasons guide
```

</div>

### Data Fields

| Field | Description | Example |
|-------|-------------|---------|
| `Category` | Main category | Tourist Spots, Food, Hotels, Shopping |
| `Sub_Category` | Specific type | Beach, Temple, Biryani, Heritage |
| `Name` | Place name | Taj Mahal, Paradise Restaurant |
| `Location_Area` | Location within city | Taj Ganj, Charminar |
| `Price_Range_INR` | Cost range | ₹500-1000, Free |
| `Description` | Detailed description | UNESCO World Heritage white marble mausoleum |
| `Rating` | User rating | 4.8/5 |
| `Contact_Timings` | Operating hours | 6AM-6:30PM |

<br/>

## 📁 Project Structure

```
🌍 Travel Planner/
│
├── 📂 Code/
│   ├── 🐍 app.py              # Streamlit web application
│   ├── 🎓 train_model.py      # Model training script
│   ├── 📊 data.csv            # Training data
│   └── 📂 model/
│       ├── 🧠 travel_model.pt     # Trained PyTorch model
│       ├── 📝 vectorizer.pkl      # TF-IDF vectorizer
│       └── 🏷️ label_encoder.pkl   # Label encoder
│
├── 📂 Dataset/
│   ├── 📊 10 cities.csv       # Master dataset
│   ├── 📊 [City].csv          # Individual city datasets
│   └── 📊 SEASONS.csv         # Seasonal travel guide
│
└── 📄 README.md
```

<br/>

## 🎨 Sample Queries

Try these queries to explore our recommendations:

| Query | What You'll Get |
|-------|----------------|
| *"ancient temples with spiritual significance"* | Kashi Vishwanath Temple, Kapaleeshwarar Temple |
| *"best street food experience"* | Kachori Gali, Gokul Chat, Deena Chat Bhandar |
| *"luxury hotels with lake views"* | Taj Lake Palace, The Oberoi Udaivilas |
| *"UNESCO World Heritage sites"* | Taj Mahal, Mahabalipuram, Agra Fort |
| *"beaches with nightlife"* | Baga Beach, Anjuna Beach, Calangute Beach |
| *"authentic biryani restaurants"* | Paradise Restaurant, Bawarchi, Shah Ghouse |

<br/>

## 🛠️ Tech Stack

<div align="center">

| Category | Technology |
|----------|------------|
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) |
| **ML Framework** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) |
| **NLP** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) |
| **Web App** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) |
| **Data** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) |

</div>

<br/>

## 📈 Model Performance

The model is trained for **1000 epochs** with Adam optimizer:

- **Input Features**: TF-IDF vectors from place descriptions
- **Output Classes**: 407 unique places
- **Hidden Layer**: 128 neurons with ReLU activation
- **Loss Function**: CrossEntropyLoss

<br/>

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔃 Open a Pull Request

<br/>

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<br/>

---

<div align="center">

### 🌟 Star this repo if you found it helpful!

Made with ❤️ for travelers exploring India

**[⬆ Back to Top](#-travelbot-ai)**

</div>
