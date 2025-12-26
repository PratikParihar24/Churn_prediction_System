import joblib
import pandas as pd
import numpy as np

def make_prediction(input_data):
    # 1. Load the artifacts
    model = joblib.load('models/churn_rf_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # 2. Get the list of columns the model was trained on
    model_columns = model.feature_names_in_
    
    # 3. Create a DataFrame with all zeros
    df_input = pd.DataFrame(columns=model_columns)
    df_input.loc[0] = 0
    
    # 4. Fill in the numerical values
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in num_cols:
        df_input[col] = input_data[col]
        
    # 5. Fill in the binary/categorical values
    # We map the user's choice to the corresponding '1' in our dummy columns
    for key, value in input_data.items():
        if key not in num_cols:
            col_name = f"{key}_{value}"
            if col_name in model_columns:
                df_input[col_name] = 1
    
    # 6. Scaling numerical columns
    df_input[num_cols] = scaler.transform(df_input[num_cols])
    
    # 7. Final Prediction
    prob = model.predict_proba(df_input)[0][1]
    prediction = model.predict(df_input)[0]
    
    return prediction, prob