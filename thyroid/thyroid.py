import pandas as pd
import numpy as np
import serial
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics

# Serial Port Setup
ser = serial.Serial(port="COM4", baudrate=9600, timeout=0.5)

print("KNN Algorithm Process")

# Load Dataset
dataset1 = pd.read_csv("DATA.csv")  # Ensure your file has B,C,T,H,G,y,P columns

# Inputs (first 5 columns)
x = dataset1.iloc[:, :-2].values
# Outputs (last 2 columns)
y = dataset1.iloc[:, -2:].values

print("x =", x)
print("y =", y)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Model setup
knn_base = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn = MultiOutputClassifier(knn_base)

# Train model
knn.fit(x_train, y_train)

# Predict and print accuracy
y_pred = knn.predict(x_test)
print("y_pred =", y_pred)
print("Testing Accuracy:")
print("Output y Accuracy:", metrics.accuracy_score(y_test[:, 0], y_pred[:, 0]))
print("Output P Accuracy:", metrics.accuracy_score(y_test[:, 1], y_pred[:, 1]))

print("Serial Enabled")

while True:
    try:
        a = ser.readline().decode('ascii').strip()
        if not a:
            continue

        print("Received:", a)

        # Default values
        a1 = a2 = a3 = a4 = a5 = 0

        if 'B' in a:
            a1 = int(a[a.index('B')+1 : a.index('B')+4])
            print("HEART RATE VALUE :", a1)
        if 'C' in a:
            a2 = int(a[a.index('C')+1 : a.index('C')+4])
            print("SPO2 VALUE :", a2)
        if 'T' in a:
            a3 = int(a[a.index('T')+1 : a.index('T')+4])
            print("TEMPERATURE VALUE :", a3)
        if 'H' in a:
            a4 = int(a[a.index('H')+1 : a.index('H')+4])
            print("HUMIDITY VALUE :", a4)
        if 'E' in a:
            a5 = int(a[a.index('E')+1 : a.index('E')+4])
            print("GLUCOSE VALUE :", a5)

        input_features = [[a1, a2, a3, a4, a5]]

        # Get predictions and probabilities
        prediction = knn.predict(input_features)[0]
        probabilities = knn.predict_proba(input_features)

        prob_y = probabilities[0][0] * 100  # for y
        prob_p = probabilities[1][0] * 100  # for P

        print(f"Predicted Outputs:  {prediction[0]},\n Percentage = {prediction[1]}%")
        print(f"Probability for y (NORMAL): {prob_y[1]:.2f}%, ABNORMAL: {prob_y[0]:.2f}%")

    except Exception as e:
        print("Error:", e)
