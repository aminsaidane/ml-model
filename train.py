import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('data.csv')

X = df.drop(columns=['resource_id'])
y = df['resource_id']

# Flatten any list-type columns
for col in ['requiredCompetences', 'requiredCertifications', 'requiredFormations']:
    if col in X.columns:
        X[col] = X[col].fillna('').apply(lambda x: eval(x) if isinstance(x, str) else x)
        mlb = MultiLabelBinarizer()
        encoded = pd.DataFrame(mlb.fit_transform(X[col]), columns=[f"{col}_{cls}" for cls in mlb.classes_])
        X = pd.concat([X.drop(columns=[col]), encoded], axis=1)

# Fill or encode remaining
X = pd.get_dummies(X.fillna('missing'))

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'model.pkl')
print("[✔] Model trained and saved as model.pkl")
