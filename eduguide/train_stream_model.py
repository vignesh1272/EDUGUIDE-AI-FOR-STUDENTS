import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ========================
# Load dataset
# ========================
data = pd.read_csv('data/IIE.csv')   # Adjust path if needed

# ========================
# Features and target
# ========================
# Features used by the model (must match order in app.py)
features = ['Maths', 'Science', 'Social', 'English', 'Interest_Encoded']
X = data[features]

# Target
y = data['Recommended_Group']

# ========================
# Encode target
# ========================
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# ========================
# Train model (Decision Tree)
# ========================
model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# ========================
# Fit an encoder for interest names
# (maps user‑selected interest name to the encoded value)
# ========================
interest_encoder = LabelEncoder()
interest_encoder.fit(data['Interest'])   # Uses the original interest names

# ========================
# Save all files
# ========================
joblib.dump(model, 'models/streams_model.pkl')
joblib.dump(target_encoder, 'models/target_encoder.pkl')
joblib.dump(interest_encoder, 'models/interest_encoder.pkl')

print("All school model files saved:")
print(" - models/streams_model.pkl")
print(" - models/target_encoder.pkl")
print(" - models/interest_encoder.pkl")

