import joblib
import pandas as pd

model = joblib.load('model.pkl')

# Example input (simulate new task)
sample = pd.DataFrame([{
    'name': 'Replace meter',
    'duration': 2,
    'businessUnit': 'Maintenance',
    'profile': 'Technician',
    'requiredCompetences': ['Electrical'],
    'requiredCertifications': ['LowVoltage'],
    'requiredFormations': ['Safety']
}])

# Same preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
for col in ['requiredCompetences', 'requiredCertifications', 'requiredFormations']:
    sample[col] = sample[col].apply(lambda x: x if isinstance(x, list) else [])

    mlb = MultiLabelBinarizer()
    sample = sample.join(
        pd.DataFrame(mlb.fit_transform(sample[col]), columns=[f"{col}_{c}" for c in mlb.classes_])
    ).drop(columns=col)

sample = pd.get_dummies(sample)

# Match input to training columns (if needed)
trained_columns = model.feature_names_in_
for col in trained_columns:
    if col not in sample.columns:
        sample[col] = 0
sample = sample[trained_columns]

pred = model.predict(sample)
print(f"[✔] Predicted resource ID: {pred[0]}")
