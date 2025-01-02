from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
import joblib
import shap
import numpy as np
import pandas as pd
import io
from io import StringIO

# Create FastAPI app instance
app = FastAPI()

# Load your trained ML model
with open("model_RandomForest.joblib", "rb") as f:
    trained_model = joblib.load(f)

# Define the request body using Pydantic
class PredictionRequest(BaseModel):
    Pregnancies: float
    Glucose: float
    Blood_Pressure: float
    Skin_Thickness: float
    Insulin: float
    BMI: float
    Diabetes_Pedigree_Function: float
    Age: float

# Define prediction endpoint for single prediction
@app.post("/predictSingle")
async def predictSingle(data: PredictionRequest=None):
    # If data is provided
    if data:
        # Convert input data to numpy array
        input_data = np.array([[
            data.Pregnancies, data.Glucose, data.Blood_Pressure, data.Skin_Thickness,
            data.Insulin, data.BMI, data.Diabetes_Pedigree_Function, data.Age
        ]])
        
        # Make prediction
        prediction = trained_model.predict(input_data)
        prediction_prob = trained_model.predict_proba(input_data)
        result = "Diabetic" if int(prediction[0]) == 1 else "Non-Diabetic"
        pred_prob = prediction_prob[0][0] if int(prediction[0]) == 0 else prediction_prob[0][1]
        explainer = shap.TreeExplainer(trained_model)
        shap_values = explainer.shap_values(input_data)
        values_ = []
        for i in range(len(shap_values[0])):
            values_.append(abs(shap_values[0][i][0]))
        return {"prediction":result, "pred_prob":pred_prob, "values_":values_}
    
# Define prediction endpoint for batch prediction
@app.post("/predictBatch")
async def predictBatch(file: UploadFile = None):
    # If file is uploaded
    if file:
        try:
            # Read the uploaded file
            content = await file.read()
            batch_df = pd.read_csv(io.BytesIO(content))

            # Check for correct columns (optional)
            input_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            if not all(col in batch_df.columns for col in input_features):
                raise HTTPException(status_code=400, detail="Invalid CSV format")

            # Predict for batch data
            predictions = trained_model.predict(batch_df[input_features])
            batch_df["prediction"] = ["Diabetic" if p == 1 else "Non-Diabetic" for p in predictions]

            # Convert the DataFrame to a CSV string
            output = StringIO()
            batch_df.to_csv(output, index=False)
            return {"batch_pred": output.getvalue()}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")