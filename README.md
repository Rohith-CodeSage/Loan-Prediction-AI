# 🧠 Personal Loan Prediction using Artificial Neural Network

This project leverages deep learning (ANN) to predict the approval of personal loans based on financial and demographic attributes. It aims to assist banks and financial institutions in making smarter, more accurate loan decisions.

---

## 📌 Project Objective

To automate and improve loan approval processes by developing a high-accuracy prediction system using Artificial Neural Networks.

---

## 🏗️ Model Architecture

- **Input Features:**  
  Age, Income, Credit Score, Expenses, Employment, Residence Status, Existing Loans, Loan Amount  
  + custom engineered feature: `ApprovalBoost`

- **Architecture:**
  - Input Layer → 14 Neurons
  - Hidden Layer 1 → 128 Neurons (ReLU)
  - Hidden Layer 2 → 64 Neurons (ReLU)
  - Hidden Layer 3 → 32 Neurons (ReLU)
  - Output Layer → 1 Neuron (Sigmoid)

- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy, Precision, Recall, F1-Score

> ✅ Final Accuracy: **91.3%**

---

## 👨‍🏫 Guided By

**Dr. V. Venkata Ramana**  
Professor, SR University  
[Google Scholar Profile](https://scholar.google.co.in/citations?user=u1Bs-GsAAAAJ&hl=en)

---

## 👨‍💻 Team Members

| Name             | Hall Ticket Number |
|------------------|--------------------|
| U. Rohith        | 2303A52198         |
| B. Rithwik       | 2303A52330         |
| K. Sai Teja      | 2303A52325         |
| G. Rushindhra    | 2303A52199         |

---

## 📁 Dataset

- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- Includes key financial features + engineered ApprovalBoost score

---

## 🚀 Future Scope

- Add real-time data support
- Improve interpretability (XAI techniques)
- Extend to credit cards, mortgages, insurance approvals
- Integrate with hybrid ML + DL models

---

## 📚 References

- Deep Learning by Ian Goodfellow
- Keras Documentation
- Financial ML research papers listed in project report

---

> _This project showcases the powerful role of deep learning in transforming modern finance._
