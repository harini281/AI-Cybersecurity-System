import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_excel("GREYC-NISLABKeystrokeBenchmarkDatasetSyed.xlsx")

# Clean columns
data.columns = data.columns.str.strip()

# Convert string → list
data["Keystroke Template Vector"] = data["Keystroke Template Vector"].apply(
    lambda x: [float(i) for i in str(x).split()]
)

# Expand vector into features
vector_df = pd.DataFrame(data["Keystroke Template Vector"].tolist())

# Features & target
X = vector_df
y = data["Class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))