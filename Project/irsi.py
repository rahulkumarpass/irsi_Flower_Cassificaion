# Iris Flower Classification using Kaggle CSV Dataset

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# 2. Load the Dataset
df = pd.read_csv("IRIS.csv")   # <-- Change filename if needed

print("\n✅ First 5 rows of dataset:")
print(df.head())

# 3. Data Information
print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Summary Statistics:")
print(df.describe())

# Check for missing values
print("\n🔍 Missing Values:")
print(df.isnull().sum())

# 4. Exploratory Data Analysis (EDA)
sns.pairplot(df, hue='species')
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 5. Data Preprocessing
# Encode species column (convert from text to numbers)
label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])

# Split features and target
X = df.drop('species', axis=1)
y = df['species']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split into Training & Test Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Model Training
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

print("\n🤖 Model Training & Accuracy Results:")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.2f}")

# 8. Evaluate the Best Model (Example: Random Forest)
best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\n🧾 Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9. Hyperparameter Tuning (Optional)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [2, 4, 6, None],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)

print("\n🏆 Best Parameters from Grid Search:")
print(grid_search.best_params_)

# 10. Save the Model
joblib.dump(grid_search.best_estimator_, "iris_model.pkl")
print("\n💾 Model saved as iris_model.pkl")

# 11. Predict on New Data
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_scaled = scaler.transform(sample)
prediction = grid_search.best_estimator_.predict(sample_scaled)
species_name = label_encoder.inverse_transform(prediction)
print("\n🌸 Prediction for sample [5.1, 3.5, 1.4, 0.2]:", species_name[0])
