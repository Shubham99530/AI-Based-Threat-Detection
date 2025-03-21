# AI-Based-Threat-Detection
# 📊 Exploratory Data Analysis (EDA) & Deep Learning Classification

This project combines **Exploratory Data Analysis (EDA)** with **Neural Network-based classification** to extract insights and build predictive models from a dataset. It involves feature engineering, PCA-based dimensionality reduction, and training deep learning models using TensorFlow/Keras.

---

## 📁 Project Structure

project.ipynb # Main Jupyter Notebook with EDA and Deep Learning workflow README.md # Project overview and documentation
---

## 🔍 Key Concepts Covered

### 🧪 Exploratory Data Analysis

- **Skewness Analysis**:
  - `|Skewness| < 1`: Approximately symmetric
  - `|Skewness| > 1`: Highly skewed
  - `1 < |Skewness| < 2`: Moderately skewed
- **Statistical Summaries**
- **Outlier Detection**
- **Visualizations**:
  - Histograms, Box Plots, Pair Plots
- **Feature Scaling** using `StandardScaler`
- **Dimensionality Reduction** using **PCA**

---

### 🤖 Deep Learning Model

- **Libraries**: `TensorFlow` / `Keras`
- **Model Architecture**:
  - Feedforward Neural Networks (Multilayer Perceptron)
  - Input: Scaled and PCA-reduced features
  - Output: One-hot encoded target variable
- **Training Details**:
  - Loss function: Categorical Crossentropy
  - Optimizer: Adam
  - Epochs: 100
  - Early stopping & checkpoint callbacks used
- **Evaluation Metrics**:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1)

---

## 📊 Example Results

- **Training Accuracy**: ✅ (Check notebook for exact value)
- **Validation Accuracy**: ✅ (Plotted using `history`)
- **Confusion Matrix**: ✔️ Reveals class-wise performance
- **PCA Explained Variance**: ✔️ Helps understand dimensionality contribution

---

## 🛠️ Tech Stack

- Python 3.x
- Jupyter Notebook
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `tensorflow` / `keras`

---

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/eda-deeplearning-project.git
   cd eda-deeplearning-project
2. **Set Up Virtual Environment**

	```bash
	python -m venv venv
	source venv/bin/activate  # Windows: venv\Scripts\activate
	pip install -r requirements.txt
3. **Run the Notebook**

	```bash
	jupyter notebook project.ipynb
---
## ✅ Outcomes
- Insights on data distribution and skewness
- PCA-based dimensionality reduction
- Trained deep learning model with high accuracy
- Clear evaluation via confusion matrix and classification report
## 📌 Notes
- Ensure the dataset is placed correctly or loaded within the notebook.
- You can tune the model architecture, learning rate, or add regularization for improvement.
