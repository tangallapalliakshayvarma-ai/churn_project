Niceee, good job getting it deployed! 🥳
Here’s a **ready-to-paste `README.md`** for your GitHub repo, including the **live Streamlit link**.

You can just copy everything below and replace your current README on GitHub.

---

````markdown
# 🧠 Customer Churn Prediction using ANN (Streamlit + Power BI)

This project predicts whether a **bank customer will churn (leave the bank)** using an  
**Artificial Neural Network (ANN)** model built in TensorFlow/Keras, and deploys it as:

- 🌐 An interactive **Streamlit web app**
- 📊 A **Power BI dashboard** for business analysis

---

## 🔗 Live Demo

🚀 **Streamlit App:**  
https://portfolio-fx5wudr4gvjrsddtnvuvsz.streamlit.app/

(Open on mobile or desktop, enter customer details, and see churn probability in real time.)

---

## 📂 Project Structure

```bash
churn_project/
│
├── app.py                          # Streamlit web app for live churn prediction
├── powerbi.py                      # Script to generate predictions CSV for Power BI
├── Churn_Modelling.csv             # Original churn dataset (10,000 customers)
├── Churn_Dashboard.pbix            # Power BI dashboard file
│
├── ann_model.h5                    # Trained ANN model
├── scaler.pk1                      # StandardScaler for feature scaling
├── label_encoder_gender.pk1        # LabelEncoder for Gender
├── onehot_encoder_geo.pk1          # OneHotEncoder for Geography
│
├── experiment.ipynb                # Model training / experimentation notebook
├── prediction.ipynb                # Prediction / evaluation notebook
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
````

---

## 🎯 Project Objective

Banks lose a lot of revenue when customers close their accounts (**churn**).
This project aims to:

> **Predict the probability that a customer will churn, so the bank can take retention actions in advance.**

---

## 📊 Dataset Description

Dataset: `Churn_Modelling.csv`
Rows: **10,000 customers**

Key columns:

| Column            | Description                               |
| ----------------- | ----------------------------------------- |
| `CustomerId`      | Unique ID for each customer               |
| `Geography`       | Country (France, Spain, Germany)          |
| `Gender`          | Male / Female                             |
| `Age`             | Age of customer                           |
| `Tenure`          | Years the customer stayed with the bank   |
| `Balance`         | Account balance                           |
| `NumOfProducts`   | Number of bank products used              |
| `HasCrCard`       | 1 = Has credit card, 0 = No credit card   |
| `IsActiveMember`  | 1 = Active customer, 0 = Inactive         |
| `EstimatedSalary` | Estimated yearly salary                   |
| `Exited`          | **Target** → 1 = Churned, 0 = Not churned |

---

## 🧠 Model Overview (ANN)

The ANN model is built using **TensorFlow/Keras**.

* Input: engineered numeric features (after encoding + scaling)
* Hidden Layers: Dense layers with **ReLU** activation
* Output Layer: **Sigmoid** activation → returns churn probability between **0 and 1**

Prediction rule:

* `probability > 0.5` → **Customer likely to churn**
* `probability ≤ 0.5` → **Customer likely to stay**

The trained model is stored in:

* `ann_model.h5`
* With preprocessing objects:

  * `scaler.pk1`
  * `label_encoder_gender.pk1`
  * `onehot_encoder_geo.pk1`

---

## 🌐 Streamlit App (`app.py`)

The Streamlit app allows users to enter customer details and get a real-time churn prediction.

### 🔹 Features

* User-friendly UI with dropdowns, sliders, and number inputs
* Inputs:

  * Geography, Gender, Age, Tenure, Balance, Credit Score, etc.
* Outputs:

  * **Churn Probability**
  * Clear message:

    * 🟢 “Customer is not likely to churn”
    * 🔴 “Customer is likely to churn”

### 🏃‍♂️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/tangallapalliakshayvarma-ai/churn_project.git
cd churn_project

# 2. (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate   # On Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Streamlit app
streamlit run app.py
```

The app will open at:
`http://localhost:8501`

---

## 📊 Power BI Integration (`powerbi.py`)

For analytics and visualization, predictions are exported to a CSV file and loaded into Power BI.

### 🔹 Steps

1. Run the script:

```bash
python powerbi.py
```

2. This generates:

```text
ann_churn_predictions_for_powerbi.csv
```

with columns like:

* `CustomerId`
* `Geography`
* `Gender`
* `Tenure`
* `Balance`
* `EstimatedSalary`
* `Actual_Churn_Status`
* `Churn_Probability`
* `Predicted_Churn_Class`

3. Open `Churn_Dashboard.pbix` in **Power BI Desktop**
4. Connect / refresh the data source to use `ann_churn_predictions_for_powerbi.csv`
5. Explore:

* Churn by Geography / Gender / Age
* Actual vs Predicted churn
* Churn probability distribution

---

## 🛠 Tech Stack

* **Language:** Python
* **ML / DL:** TensorFlow, Keras, scikit-learn
* **Frontend:** Streamlit
* **Data Handling:** pandas, numpy
* **Visualization:** Power BI

---

## 🚀 Deployment

* **Platform:** Streamlit Community Cloud
* **App URL:**
  👉 [https://portfolio-fx5wudr4gvjrsddtnvuvsz.streamlit.app/](https://portfolio-fx5wudr4gvjrsddtnvuvsz.streamlit.app/)

To deploy your own fork:

1. Fork this repository on GitHub
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Create a new app → connect your forked repo
4. Select:

   * Repo: `your-username/churn_project`
   * Branch: `main`
   * Main file: `app.py`
5. Deploy and get your own public URL 🎉

---

## 📌 Future Enhancements

* Add more countries and retrain on real Indian customer data
* Use **Explainable AI (LIME/SHAP)** to show which features influenced churn
* Add authentication to restrict access to business users
* Containerize with Docker and deploy on cloud (AWS/Azure/GCP)

---

## 🙌 Acknowledgements

* Dataset inspired by popular **Customer Churn Modelling** datasets used in ML tutorials.
* Built as an educational project to demonstrate **end-to-end ML deployment**:

  * Data → Model → API/UI → Dashboard.

