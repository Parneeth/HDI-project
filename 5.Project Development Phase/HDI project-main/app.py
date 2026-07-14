import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
import logging
import pickle

from utils.preprocessing import load_and_preprocess_data
from utils.predictor import HDIPredictionEngine
from utils.helper import calculate_summary_statistics, get_suggested_insights

# Initialize Flask app
app = Flask(__name__, static_folder="dist", template_folder="templates")

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
CSV_PATH = os.path.join(os.getcwd(), "dataset", "HDI.csv")
MODEL_DIR = os.path.join(os.getcwd(), "model")
MODEL_PKL = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PKL = os.path.join(MODEL_DIR, "scaler.pkl")

# Initialize Global Models
os.makedirs(MODEL_DIR, exist_ok=True)
predictor_engine = None

def init_predictor():
    global predictor_engine
    if os.path.exists(MODEL_PKL) and os.path.exists(SCALER_PKL):
        predictor_engine = HDIPredictionEngine(MODEL_PKL, SCALER_PKL)
        logger.info("Python Predictor Engine successfully initialized with pickle weights.")
    else:
        logger.warning("No pickle weights found. Predictor engine initialized in heuristic fallback mode.")
        predictor_engine = HDIPredictionEngine(None, None)

@app.route("/api/dataset", methods=["GET"])
def get_dataset():
    """
    Returns descriptive dataset summary information and raw records.
    """
    try:
        if not os.path.exists(CSV_PATH):
            return jsonify({"error": "Dataset HDI.csv not found in database folder."}), 404
        
        df = pd.read_csv(CSV_PATH)
        
        # Calculate summary statistics
        summary_dict = calculate_summary_statistics(df)
        summary_list = []
        for col, stats in summary_dict.items():
            stats["column"] = col
            summary_list.append(stats)

        # Get raw records limit to first 150 for performance
        records = df.to_dict(orient="records")
        
        return jsonify({
            "totalRows": len(df),
            "columns": list(df.columns),
            "duplicates": int(df.duplicated().sum()),
            "summary": summary_list,
            "rows": records
        })
    except Exception as e:
        logger.error(f"Error serving dataset API: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/eda", methods=["GET"])
def get_eda():
    """
    Precomputes correlations and distributions for the frontend to render immediately.
    """
    try:
        if not os.path.exists(CSV_PATH):
            return jsonify({"error": "Dataset HDI.csv not found."}), 404
            
        df = pd.read_csv(CSV_PATH)
        numeric_cols = ["Life_Expectancy", "Expected_Schooling", "Mean_Schooling", "GNI_Per_Capita", "HDI"]
        
        # 1. Pearson Correlation Cell Grid
        corr = df[numeric_cols].corr()
        correlations_list = []
        for col_x in numeric_cols:
            for col_y in numeric_cols:
                correlations_list.append({
                    "x": col_x,
                    "y": col_y,
                    "value": float(corr.at[col_x, col_y])
                })
                
        # 2. Histogram Bins
        histograms_dict = {}
        for col in numeric_cols:
            series = df[col].dropna()
            counts, bin_edges = np.histogram(series, bins=10)
            bins_list = []
            for i in range(len(counts)):
                bins_list.append({
                    "binStart": float(bin_edges[i]),
                    "binEnd": float(bin_edges[i+1]),
                    "label": f"{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}",
                    "count": int(counts[i])
                })
            histograms_dict[col] = bins_list

        # 3. Scatter Plot Points
        scatter_points = []
        for idx, row in df.iterrows():
            scatter_points.append({
                "country": row["Country"],
                "lifeExpectancy": float(row["Life_Expectancy"]),
                "expectedSchooling": float(row["Expected_Schooling"]),
                "meanSchooling": float(row["Mean_Schooling"]),
                "gniPerCapita": float(row["GNI_Per_Capita"]),
                "hdi": float(row["HDI"])
            })

        return jsonify({
            "correlations": correlations_list,
            "histograms": histograms_dict,
            "scatterData": scatter_points
        })
    except Exception as e:
        logger.error(f"Error serving EDA API: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/train", methods=["POST"])
def train_model():
    """
    Trains/tunes four regression algorithms, selects the best model, and dumps pickles.
    """
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        if not os.path.exists(CSV_PATH):
            return jsonify({"error": "Socioeconomic dataset (HDI.csv) is missing."}), 404

        # Run preprocessing train/test split
        X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess_data(CSV_PATH)

        # Regressors
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(max_depth=4, min_samples_split=5, random_state=42),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=15, max_depth=4, min_samples_split=5, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=15, learning_rate=0.1, max_depth=3, random_state=42)
        }

        metrics_results = {}
        best_r2 = -float("inf")
        best_name = None
        best_estimator = None

        for name, clf in models.items():
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            r2 = float(r2_score(y_test, preds))
            mae = float(mean_absolute_error(y_test, preds))
            mse = float(mean_squared_error(y_test, preds))
            rmse = float(np.sqrt(mse))

            metrics_results[name] = {
                "modelName": name,
                "metrics": {
                    "r2": r2,
                    "mae": mae,
                    "mse": mse,
                    "rmse": rmse
                },
                "predictions": list(preds[:25]),  # Sample predictions
                "residuals": list((y_test - preds)[:25])
            }

            if r2 > best_r2:
                best_r2 = r2
                best_name = name
                best_estimator = clf

        # Serialize best performing weights
        with open(MODEL_PKL, "wb") as f:
            pickle.dump(best_estimator, f)
        with open(SCALER_PKL, "wb") as f:
            pickle.dump(scaler, f)

        # Precompute relative feature importance profiles
        feature_importance_list = []
        if hasattr(best_estimator, "feature_importances_"):
            importance = best_estimator.feature_importances_
            for i, feat in enumerate(features):
                feature_importance_list.append({
                    "feature": feat,
                    "importance": float(importance[i])
                })
        else:
            # Linear coefficients fallback profile
            coefs = np.abs(best_estimator.coef_)
            total_coef = np.sum(coefs) if np.sum(coefs) > 0 else 1.0
            for i, feat in enumerate(features):
                feature_importance_list.append({
                    "feature": feat,
                    "importance": float(coefs[i] / total_coef)
                })

        # Re-init prediction engine with new serialized objects
        init_predictor()

        return jsonify({
            "bestModel": best_name,
            "metrics": metrics_results,
            "featureImportance": feature_importance_list,
            "testSetCount": len(y_test),
            "serialized": True
        })
    except Exception as e:
        logger.error(f"Error during training pipeline: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts customized parameters, generates the HDI prediction score, and returns AI insights.
    """
    global predictor_engine
    try:
        data = request.get_json() or {}
        
        # Validations
        country = data.get("country", "").strip()
        if not country:
            return jsonify({"error": "Country parameter is missing."}), 400
            
        try:
            life_exp = float(data.get("lifeExpectancy"))
            exp_sch = float(data.get("expectedSchooling"))
            mean_sch = float(data.get("meanSchooling"))
            gni_capita = float(data.get("gniPerCapita"))
        except (TypeError, ValueError):
            return jsonify({"error": "All socioeconomic parameters must be numeric values."}), 400

        if not (20 <= life_exp <= 100):
            return jsonify({"error": "Life expectancy must reside within [20, 100] years."}), 400
        if not (0 <= exp_sch <= 30) or not (0 <= mean_sch <= 30):
            return jsonify({"error": "Schooling figures must reside within [0, 30] years."}), 400
        if not (100 <= gni_capita <= 200000):
            return jsonify({"error": "Gross National Income must reside within [$100, $200,000] per capita."}), 400

        # Lazy init if needed
        if predictor_engine is None:
            init_predictor()

        # Run Prediction
        score, category = predictor_engine.predict(life_exp, exp_sch, mean_sch, gni_capita)

        # Get Policy Recommendations & Insights
        insights_data = get_suggested_insights(country, score, category)
        
        insights_md = f"""### Socioeconomic Analysis for **{country}**
The predicted HDI score is **{score:.3f}**, which classifies this country as a **{category} Human Development** nation.

### Key Bottleneck Analysis
* The primary developmental challenge detected for {country} involves **{insights_data['bottlenecks']}**. 
* {insights_data['evaluation']}

### Strategic Policy Recommendations
1. **{insights_data['recommendations'][0]}**
2. **{insights_data['recommendations'][1]}**"""

        return jsonify({
            "country": country,
            "lifeExpectancy": life_exp,
            "expectedSchooling": exp_sch,
            "meanSchooling": mean_sch,
            "gniPerCapita": gni_capita,
            "predictedScore": score,
            "formattedScore": f"{score:.4f}",
            "predictedCategory": category,
            "insights": insights_md
        })
    except Exception as e:
        logger.error(f"Error during predictive inference: {e}")
        return jsonify({"error": str(e)}), 500

# Server Static Files & SPA Routing
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    init_predictor()
    app.run(host="0.0.0.0", port=3000)
