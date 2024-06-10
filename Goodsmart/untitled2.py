# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 06:54:42 2024

@author: youss
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , accuracy_score,roc_auc_score, roc_curve, confusion_matrix,precision_recall_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns

df = pd.read_csv('goodsmart.csv')
df.head()
df.shape
df['Yes or No'].value_counts()
df

#When axis=0, it refers to operations along the rows 
#When axis=1, it refers to operations along the columns 

# Drop the first column (ID column)
df = df.drop(df.columns[0], axis=1)
df

# Select all columns except the first as the feature set
X = df.iloc[:, 1:]
# Assign the first column as the target variable
y = df.iloc[:, 0]

#convert these to numpy arrays
feature_set_array = X.values
target_variable_array = y.values
feature_set_array[0:4]
y

#fit(): This method computes the mean and standard deviation of each feature in the dataset 
#transform(): This method transforms the dataset using the computed mean and standard deviation.
# Scale the feature set
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train , X_test , y_train , y_test = train_test_split(X_scaled , y , test_size=0.2 , random_state=4)

# gamma dah el byhded el curveture bta3 el khat ely hyfsel maben el classes

classifier = svm.SVC (kernel='linear' , gamma='auto', C=2)
classifier.fit(X_train , y_train)

y_predict = classifier.predict(X_test)
y_predict

print(classification_report(y_test , y_predict))

# Accuracy
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy:", accuracy)

#fit_transform() During the fitting step, the method learns parameters from the data (like the mean and standard deviation for scaling
# or the mapping between categories and numerical values for label encoding).
# Convert categorical labels to numerical labels
label_encoder = LabelEncoder()
y_test_numeric = label_encoder.fit_transform(y_test)

#transofrm It's used when you have already fitted a transformation on a training dataset
# Convert predicted labels to numerical labels
y_predict_numeric = label_encoder.transform(y_predict)


# # ROC Curve and AUC
# roc_auc = roc_auc_score(y_test_numeric, y_predict_numeric)
# print("ROC AUC Score:", roc_auc)

# fpr, tpr, _ = roc_curve(y_test_numeric, y_predict_numeric)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print(conf_matrix)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test_numeric, y_predict_numeric)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# def create_gui():
#     window = tk.Tk()
#     window.title("Data Input Form")

#     # Padding for the entire application
#     app_padding = 20

#     # List of labels for the input fields
#     labels = [
#         "Rehab_Num", "B_Num", "perimeter_mean", "Group", "Building_Num",
#         "Apartment_Num"
#     ]

#     # Create a dictionary to hold the Entry widgets
#     entries = {}
    
   #  # Define dimensions
   # input_width = 6  # Approximate width for the input fields
   # num_columns = 3
   # spacing = 10

   # # Create Frame for the first row (input fields and labels)
   # input_frame = tk.Frame(window)
   # input_frame.pack(side=tk.TOP, pady=app_padding, padx=app_padding)

   # # Create labels and entry fields in a grid for the first row
   # for index, label_text in enumerate(labels):
   #     row = index // num_columns
   #     column = index % num_columns
   #     label = tk.Label(input_frame, text=label_text)
   #     label.grid(row=row, column=column*2, sticky=tk.W, padx=spacing, pady=spacing)

   #     entry = tk.Entry(input_frame, width=input_width)
   #     entry.grid(row=row, column=column*2+1, padx=spacing, pady=spacing)
   #     entries[label_text] = entry

   # # Submit button
   # submit_button = tk.Button(window, text="Submit", command=lambda: submit(entries))
   # submit_button.pack(side=tk.BOTTOM, pady=app_padding, padx=app_padding)  # Padding added to top, bottom, left, and right

   # window.mainloop()

# def submit(entries):
#     # Function to handle submission of data
#     data_list = []
#     for label, entry in entries.items():
#         value = entry.get().strip()  # Get the entry value and remove leading/trailing whitespace
#         if not value:
#             # If the entry is empty
#             messagebox.showerror("Error", "Please fill all empty cells")
#             return  # Exit the function early if any error occurs
#         try:
#             value = float(value)  # Try converting to float
#             data_list.append(value)
#         except ValueError:
#             # If conversion to float fails, it's not a valid numeric value
#             messagebox.showerror("Error", "Please enter valid numeric values")
#             return  # Exit the function early if any error occurs
#     print("Data submitted as list:", data_list)
#     # Continue with the rest of your code (scaling and prediction)
#     scaler = StandardScaler()
#     data_array = np.array(data_list).reshape(1, -1)  # Convert data list to array and reshape for scaler
#     scaled_data = scaler.fit_transform(data_array)
#     y_predict = classifier.predict(scaled_data)
    
#     # Display prediction to the user
#     messagebox.showinfo("Prediction", f"The predicted value is: {y_predict}")
    
#     create_gui()

def predict_stock_price(rehab_num, b_num, group, buld_num, apart_num, ):
    input_data = np.array([rehab_num, b_num, group, buld_num, apart_num]).reshape(1, -1)
    predicted_price = classifier.predict(input_data)
    return predicted_price[0]
# User input
user_rehab_num = float(input("Enter the rehab num: "))
user_b_num = float(input("Enter the b num: "))
user_groub = float(input("Enter the group num: "))
user_buld_num = float(input("Enter buld num: "))
user_apart_num = float(input("Enter the apartmaent num: "))
# Predict using user input
predicted_stock_price = predict_stock_price(user_rehab_num, user_b_num, user_groub, user_buld_num,user_apart_num )
print(f"Active customer or not?: {predicted_stock_price}")