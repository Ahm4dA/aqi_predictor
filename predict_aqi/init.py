import logging
import azure.functions as func
import json
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load artifacts once at cold start
scaler = joblib.load('scaler.pkl')
model  = load_model('aqi_lstm_model.h5')

def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Expects a JSON body with key "data" = list of 12 [CO, PM2.5, AQI] triples.
    Returns JSON {"forecast": [AQI_t+1, ..., AQI_t+4]}
    """
    logging.info('predict_aqi function processed a request.')

    try:
        body = req.get_json()
        data = np.array(body['data'], dtype=float)
        if data.shape != (12, 3):
            raise ValueError(f'Expected shape (12,3), got {data.shape}')
    except Exception as e:
        return func.HttpResponse(
            f"Invalid input: {e}",
            status_code=400
        )

    # Scale & reshape
    scaled = scaler.transform(data)              # (12,3)
    X_new  = np.expand_dims(scaled, axis=0)      # (1,12,3)

    # Predict next 4 AQI steps
    y_scaled = model.predict(X_new).flatten()    # (4,)

    # Inverse-scale only the AQI channel
    dummy = np.zeros((len(y_scaled), scaled.shape[1]))
    dummy[:, 2] = y_scaled
    y_pred = scaler.inverse_transform(dummy)[:, 2]

    # Return JSON
    return func.HttpResponse(
        json.dumps({'forecast': y_pred.tolist()}),
        status_code=200,
        mimetype="application/json"
    )
