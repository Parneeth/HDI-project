import pickle
import numpy as np

class HDIPredictionEngine:
    def __init__(self, model_path, scaler_path):
        """
        Initializes predictor by loading serialized pickle model and scaler.
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        except Exception as e:
            self.model = None
            self.scaler = None
            print(f"Prediction engine running in mock mode. Error loading pickles: {e}")

    def classify_hdi(self, score):
        """
        Classifies HDI score based on official UN standards.
        """
        clamped_score = max(0.0, min(1.0, score))
        if clamped_score >= 0.800:
            return clamped_score, "Very High"
        elif clamped_score >= 0.700:
            return clamped_score, "High"
        elif clamped_score >= 0.550:
            return clamped_score, "Medium"
        else:
            return clamped_score, "Low"

    def predict(self, life_exp, exp_sch, mean_sch, gni_capita):
        """
        Performs HDI prediction on a single feature vector.
        """
        input_data = np.array([[life_exp, exp_sch, mean_sch, gni_capita]])
        
        if self.model and self.scaler:
            scaled_data = self.scaler.transform(input_data)
            score = self.model.predict(scaled_data)[0]
        else:
            # Fallback high-fidelity mathematical heuristic approximating UN geometric index formula
            lei = (life_exp - 20) / (85 - 20)
            ei = ((exp_sch / 18) + (mean_sch / 15)) / 2
            ii = (np.log(gni_capita) - np.log(100)) / (np.log(75000) - np.log(100))
            score = (max(0, lei) * max(0, ei) * max(0, ii)) ** (1/3)

        return self.classify_hdi(score)
