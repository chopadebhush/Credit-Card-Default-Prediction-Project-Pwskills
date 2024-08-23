from flask import Flask, request, render_template
import os
from ccdp.pipeline.stage_06_prediction import PredictionPipeline
from ccdp.constants import MODEL_FILE_PATH, SCALER_PATH
app = Flask(__name__)

def ensure_artifacts():
    """
    Check if the model and scaler files exist.
    If not, execute main.py to generate them.
    """
    if not os.path.exists(MODEL_FILE_PATH) or not os.path.exists(SCALER_PATH):
        print("Model or Scaler not found, running main.py...")
        os.system("python main.py")  # This will run main.py to create the model and scaler

# Ensure the artifacts are available
ensure_artifacts()

# Initialize the prediction pipeline
pipeline = PredictionPipeline(model_path=MODEL_FILE_PATH, scaler_path=SCALER_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        features = [
            float(request.form['LIMIT_BAL']),
            int(request.form['SEX']),
            int(request.form['EDUCATION']),
            int(request.form['MARRIAGE']),
            int(request.form['AGE']),
            int(request.form['PAY_SEPT']),
            int(request.form['PAY_AUG']),
            int(request.form['PAY_JUL']),
            int(request.form['PAY_JUN']),
            int(request.form['PAY_MAY']),
            int(request.form['PAY_APR']),
            float(request.form['BILL_AMT_SEPT']),
            float(request.form['BILL_AMT_AUG']),
            float(request.form['BILL_AMT_JUL']),
            float(request.form['BILL_AMT_JUN']),
            float(request.form['BILL_AMT_MAY']),
            float(request.form['BILL_AMT_APR']),
            float(request.form['PAY_AMT_SEPT']),
            float(request.form['PAY_AMT_AUG']),
            float(request.form['PAY_AMT_JUL']),
            float(request.form['PAY_AMT_JUN']),
            float(request.form['PAY_AMT_MAY']),
            float(request.form['PAY_AMT_APR']),
        ]

        # Get the prediction from the pipeline
        prediction = pipeline.predict(features)

        # Display the result on the webpage
        prediction_text = f"This Credit Card holder will {'Default' if prediction == 1 else 'not Default'} next month."

    except Exception as e:
        prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
