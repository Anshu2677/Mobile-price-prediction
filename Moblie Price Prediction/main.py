import pandas as pd
# Load dataset
df = pd.read_csv("dataset/train.csv")
# Display first 5 rows
print(df.head())
# Display dataset information
print("\nDataset Shape:", df.shape)
# Display column names
print("\nColumns:")
print(df.columns.tolist())
# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Check data types
print("\nData Types:")
print(df.dtypes)

# Check target distribution
print("\nPrice Range Distribution:")
print(df["price_range"].value_counts().sort_index())
# ==============================
# MODEL TRAINING
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Features and target
X = df.drop("price_range", axis=1)
y = df["price_range"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[0, 1, 2, 3]
)

disp.plot()
plt.title("Mobile Price Range - Confusion Matrix")
plt.show()

import joblib
# Save trained model
joblib.dump(model, "models/mobile_price_model.pkl")

print("\nModel saved successfully!")
# ==============================
# USER INPUT PREDICTION
# ==============================

print("\nEnter Mobile Specifications:")

battery_power = float(input("Battery Power (mAh): "))
blue = int(input("Bluetooth (0=No, 1=Yes): "))
clock_speed = float(input("Clock Speed (GHz): "))
dual_sim = int(input("Dual SIM (0=No, 1=Yes): "))
fc = int(input("Front Camera (MP): "))
four_g = int(input("4G (0=No, 1=Yes): "))
int_memory = int(input("Internal Memory (GB): "))
m_dep = float(input("Mobile Depth (cm): "))
mobile_wt = int(input("Mobile Weight (g): "))
n_cores = int(input("Number of Cores: "))
pc = int(input("Primary Camera (MP): "))
px_height = int(input("Pixel Height: "))
px_width = int(input("Pixel Width: "))
ram = int(input("RAM (MB): "))
sc_h = int(input("Screen Height (cm): "))
sc_w = int(input("Screen Width (cm): "))
talk_time = int(input("Talk Time (hours): "))
three_g = int(input("3G (0=No, 1=Yes): "))
touch_screen = int(input("Touch Screen (0=No, 1=Yes): "))
wifi = int(input("WiFi (0=No, 1=Yes): "))

user_mobile = [[
    battery_power,
    blue,
    clock_speed,
    dual_sim,
    fc,
    four_g,
    int_memory,
    m_dep,
    mobile_wt,
    n_cores,
    pc,
    px_height,
    px_width,
    ram,
    sc_h,
    sc_w,
    talk_time,
    three_g,
    touch_screen,
    wifi
]]

prediction = model.predict(user_mobile)

print("\n==============================")
print("Predicted Price Range:", prediction[0])
print("==============================")