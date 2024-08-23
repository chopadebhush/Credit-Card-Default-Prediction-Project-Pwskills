import pickle
import numpy as np

class PredictionPipeline:
    def __init__(self, model_path, scaler_path):
        self.model = self.load_model(model_path)
        self.scaler = self.load_scaler(scaler_path)

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model

    def load_scaler(self, scaler_path):
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        return scaler

    def predict(self, features):
        features = np.array(features).reshape(1, -1)
        features = self.scaler.transform(features)
        prediction = self.model.predict(features)
        return prediction[0]

# Example usage
# pipeline = PredictionPipeline(model_path='artifacts/model.pkl', scaler_path='artifacts/scaler.pkl')
# result = pipeline.predict([240000.0, 2, 1, 1, 40, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1])
# print(f"Prediction: {result}")
