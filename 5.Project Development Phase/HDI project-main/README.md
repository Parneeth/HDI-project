# Human Development Index (HDI) Prediction System

An industry-standard, end-to-end Machine Learning web application designed to predict the Human Development Index (HDI) scores of countries based on essential socioeconomic indicators, and classify them into official UN development levels: **Very High**, **High**, **Medium**, and **Low**.

---

## 🚀 Key Features

*   **Descriptive Statistics & Data Lab**: Comprehensive data explorer showing statistical profiles (mean, standard deviation, percentiles, missing/duplicate checks) of the UN country databases.
*   **Exploratory Data Analysis (EDA)**: Interactive responsive visual charts (histograms, multi-variable correlation grids, scatter plots with dynamic tooltips) to observe indices behavior.
*   **Multi-Model Estimator Laboratory**: Comparative evaluation of 4 separate regression architectures:
    1.  **Linear Regression** (OLS Baseline)
    2.  **Decision Tree Regressor** (Pruned)
    3.  **Random Forest Regressor** (Ensemble Bagging)
    4.  **Gradient Boosting Regressor** (Ensemble Boosting)
*   **Interactive Predictor Sandbox**: Custom prediction engine featuring preloaded socioeconomic templates, real-time input validations, and custom gauge dials.
*   **AI-Powered Grounded Advisory**: Generates contextual, strategic policy analyses, bottle-neck identification, and intervention plans utilizing high-fidelity language models.

---

## 📐 Platform Architecture

```
                 +-----------------------------------------------+
                 |             User Interface Layer              |
                 |      (React 19 + Tailwind CSS + Lucide)       |
                 +-----------------------+-----------------------+
                                         |  JSON APIs
                                         v
                 +-----------------------------------------------+
                 |              Full-Stack Gateway               |
                 |        (Express.js + Node TypeScript /        |
                 |           Flask Python Backup)                |
                 +-----------------------+-----------------------+
                                         |
                                         v
                 +-----------------------------------------------+
                 |               Prediction Engine               |
                 |    (HDIPipeline + StandardScaler Preprocessor)|
                 +-----------------------+-----------------------+
                                         |  Inference
                                         v
                 +-----------------------------------------------+
                 |              Serialized ML Models             |
                 |          (Best Estimator Model JSON/PKL)      |
                 +-----------------------------------------------+
```

---

## 🛠️ Technology Stack

*   **Frontend**: React 19, TypeScript, Tailwind CSS v4, Motion (for transitions), Lucide React (for icons)
*   **Backend Server**: Node.js/Express (Active Preview runtime on port 3000) or Python/Flask (Standalone local container)
*   **Socioeconomic Analytics & ML**:
    *   **TypeScript**: Linear Regression, Scaler, Statistics helpers in `src/utils`
    *   **Python**: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
*   **Cognitive Analysis**: Google Gemini API via `@google/genai` (SDK v2)

---

## 📦 Installation & Setup

### A. Full-Stack Node.js (Vite + Express) Setup
The platform is fully optimized to compile and run on top of an Express-Vite backend integration:

1.  **Install dependencies**:
    ```bash
    npm install
    ```
2.  **Configure Environment Variables**:
    Create a `.env` file in the root directory:
    ```env
    GEMINI_API_KEY="YOUR_ACTUAL_API_KEY"
    ```
3.  **Launch the development server**:
    ```bash
    npm run dev
    ```
4.  **Compile & Build for production**:
    ```bash
    npm run build
    npm run start
    ```

### B. Python Flask Backend Setup (Alternative Local Launch)
If you prefer running a pure Python-driven stack:

1.  **Install requirements**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure environment**:
    Ensure your model is trained and weights are prepared:
    ```bash
    python -c "from utils.preprocessing import load_and_preprocess_data; load_and_preprocess_data('dataset/HDI.csv')"
    ```
3.  **Run Flask server**:
    ```bash
    python app.py
    ```

---

## 📈 Future Enhancements

*   **Explainable AI (XAI)**: Integration of SHAP (SHapley Additive exPlanations) or LIME values on the prediction results to explain individual feature impact.
*   **Geospatial Chloropleth Maps**: Render HDI score predictions across a dynamic, color-coded interactive global map.
*   **Historical Timeline Trends**: Log and graph country-specific historical HDI scores from 1990 to the present day.
