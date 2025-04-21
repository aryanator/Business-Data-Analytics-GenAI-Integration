# 📦 Business Data Analytics & GenAI Integration

A data science pipeline to investigate a 15% increase in delivery costs using business KPIs, ML modeling, hypothesis testing, and LLM-based natural language summaries.

---

## 📌 Objective

The client observed a 15% increase in delivery costs over the last 6 months while claiming that shipment volume remained stable. This project aims to:
- Verify the claim statistically
- Identify key cost drivers
- Recommend actionable steps
- Explore GenAI for executive-level insights

---

## 📊 Exploratory Data Analysis (EDA)

- **Temporal Trends**: Delivery costs peaked mid-year but declined in Q4; trend does not support a consistent 15% rise.
- **Operational Drivers**: Mileage, Fuel Used, and Driver Hours are strongly correlated with cost (r > 0.80).
- **Vendor & Regional Impact**: North region and vendors like B & G showed higher variability and medians in cost.
- **Efficiency Metrics**: Introduced KPIs like `Cost_Per_Package`, `Cost_Per_Mile`, `Cost_Per_Hour` for deeper cost diagnostics.

---

## 🤖 Modeling

### Gradient Boosting Regressor (XGBoost-style)
- **R² Score**: 0.99  
- **RMSE**: $41.67  
- **Top Predictors**:  
  - `Cost_Per_Package` (63%)  
  - `Driver_Hours` (21%)  
  - `Cost_Per_Hour` (10%)

### Neural Network (Experimental)
- **R² Score**: 0.91  
- **RMSE**: $110.8  
- Built with `Keras` for scalability but underperformed on tabular data.

---

## 📐 Statistical Testing

| Hypothesis | Test | Result |
|------------|------|--------|
| Delivery cost increased? | One-sided t-test | ❌ Not statistically significant (p=0.24) |
| Shipment volume changed? | Two-sided t-test | ❌ No significant change (p=0.58) |

---

## 🧠 GenAI Integration

### 🧩 RAG Pipeline with LLaMA 8B
- Converts structured cost logs into natural language
- Embeds data using `MiniLM` + `FAISS`
- Answers context-rich business questions like:  
  _"Why is Vendor G’s Q4 cost higher?"_

### 🔧 DistilGPT2 Fine-Tuning
- Trained on structured logs using `transformers`
- Produces explainable completions on delivery trends
- Deployed via Hugging Face inference

---

## 📈 Key Outcomes

- **No statistical basis** for cost hike; operational inefficiencies likely cause.
- **Driver Hours** and **Cost Efficiency** are actionable levers.
- **LLM-based summaries** deliver stakeholder-friendly insights.

---

## 💡 Recommendations

- Track `Cost_Per_Package` as a core KPI.
- Optimize `Driver_Hours` via route clustering.
- Reassess high-cost vendors.
- Use RAG/LLMs to auto-summarize KPI drift monthly.

---

## 📂 Files

- [`main.ipynb`](./technical_assessment.ipynb) – Full code
- [`Plots-Results.pdf`](./Plots-Results.pdf) – Visual outputs
- [`Delivery_Cost_Summary.pdf`](./Delivery_Cost_Summary.pdf) – Business report

---

## 🛠 Tech Stack

`Python` · `Pandas` · `Seaborn` · `Scikit-learn` · `Keras` · `FAISS` · `Transformers` · `Together API` · `LLaMA` · `Streamlit`
