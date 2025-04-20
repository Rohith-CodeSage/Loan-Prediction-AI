# 🧠 Personal Loan Prediction using Artificial Neural Network

---

# 👨‍🏫 Guided By

# Dr. Venkataramana Veeramsetty 
Associate Professor, Department of CS & AI  
Director, Center for AI and Deep Learning, SR University  
NVIDIA DLI Ambassador | Microsoft Azure AI Trainer  
[Google Scholar Profile](https://scholar.google.co.in/citations?user=u1Bs-GsAAAAJ&hl=en)

Dr. Veeramsetty is a leading researcher in AI, deep learning, and smart systems.  
- 🔹 600+ citations, h-index: 15, i10-index: 19  
- 🔹 Key research areas: Smart Grids, AI/ML Applications, Optimization  
- 🔹 Featured works:
  - *Short-term Electric Power Load Forecasting using RF & GRU* (2022)  
  - *Indian Currency Recognition using Deep Learning* (2020)  
  - *Electric Power Load Forecasting using ANN* (2020)

---

## 👨‍💻 Team Members

| Name             | Hall Ticket Number |
|------------------|--------------------|
| U. Rohith        | 2303A52198         |
| B. Rithwik       | 2303A52330         |
| K. Sai Teja      | 2303A52325         |
| G. Rushindhra    | 2303A52199         |

---

## 📌 Project Objective

To automate and enhance personal loan approval decisions using a deep learning model with high accuracy.

---

## 🏗️ Model Architecture

- **Input Features:**  
  Age, Income, Credit Score, Expenses, Employment, Residence Status, Existing Loans, Loan Amount  
  + custom engineered feature: `ApprovalBoost`

- **Network Design:**
  - Input Layer → 14 Neurons  
  - Hidden Layer 1 → 128 Neurons (ReLU)  
  - Hidden Layer 2 → 64 Neurons (ReLU)  
  - Hidden Layer 3 → 32 Neurons (ReLU)  
  - Output Layer → 1 Neuron (Sigmoid)

- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score  

> ✅ Final Model Accuracy: **91.3%**

---

## 📁 Dataset

- Source: [Kaggle - Loan Approval Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)  
- Enhanced with feature engineering (e.g., `ApprovalBoost`)

---

## 🚀 Future Scope

- Integrate with real-time loan application systems  
- Apply XAI (Explainable AI) for transparency  
- Extend to credit cards, mortgages, insurance domains  
- Experiment with hybrid models combining ML + DL  

---

## 📚 References

- *Deep Learning* by Ian Goodfellow  
- Official Keras & TensorFlow Documentation  
- Financial ML case studies and research papers

---

> _This project demonstrates the power of deep learning in solving real-world financial challenges._
