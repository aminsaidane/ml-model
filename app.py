from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    for col in ['requiredCompetences', 'requiredCertifications', 'requiredFormations']:
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

        mlb = MultiLabelBinarizer()
        df = df.join(
            pd.DataFrame(mlb.fit_transform(df[col]), columns=[f"{col}_{c}" for c in mlb.classes_])
        ).drop(columns=col)

    df = pd.get_dummies(df)

    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[model.feature_names_in_]

    prediction = model.predict(df)[0]
    return jsonify({"predicted_resource_id": prediction})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
