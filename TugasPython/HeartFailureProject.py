import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mengatur level logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Menonaktifkan oneDNN custom operations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import tensorflow as tf




from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import classification_report, accuracy_score
#from keras.layers import Dense, BatchNormalization, Dropout, LSTM
#from keras.models import Sequential
#from keras.optimizers import Adam
#from keras import callbacks
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score






#loading data
df = pd.read_csv(r"heart_failure.csv")
print(df.head())
print(df.info())

# Menghitung Skewness
skewness = df["DEATH_EVENT"].skew()
print(f'Skewness of DEATH_EVENT: {skewness}')

# Simpan informasi skewness ke file CSV
skewness_info = pd.DataFrame({
    'Feature': ['DEATH_EVENT'],
    'Skewness': [skewness]
})
skewness_info.to_csv('skewness_info.csv', index=False)
print("Informasi skewness telah diekspor ke skewness_info.csv")

# Visualisasi Skewness
plt.figure(figsize=(8, 6))
cols= ["#32CD32","#FF0000"]
ax = sns.countplot(x=df["DEATH_EVENT"], palette=cols)
ax.bar_label(ax.containers[0])
plt.title('Count Plot of DEATH_EVENT')
plt.xlabel('DEATH_EVENT')
plt.ylabel('Count')
plt.show()

# Deskripsi statistik
description = df.describe().T
print(description)

# Analisis Bivariat dengan matriks korelasi menggunakan heatmap
cmap = sns.diverging_palette(2, 165, s=80, l=55, n=9)
corrmat = df.corr()  # Hitung matriks korelasi
plt.subplots(figsize=(20, 20))
sns.heatmap(corrmat, cmap=cmap, annot=True, square=True)
plt.title('Heatmap of Correlation Matrix')
plt.show()

# Simpan matriks korelasi ke file CSV
corrmat.to_csv('correlation_matrix.csv')
print("Matriks korelasi telah diekspor ke correlation_matrix.csv")

# Membuat DataFrame untuk data yang dihitung
age_death_count = df.groupby(['age', 'DEATH_EVENT']).size().unstack(fill_value=0)

# Simpan DataFrame ke CSV
age_death_count.to_csv('age_death_distribution.csv')
print("Data distribusi usia berdasarkan kejadian kematian telah diekspor ke age_death_distribution.csv")

# Visualisasi distribusi usia berdasarkan kejadian kematian
plt.figure(figsize=(15, 10))
cols = ["#32CD32", "#FF0000"]  # Palet warna untuk batang
Days_of_week = sns.countplot(x=df['age'], data=df, hue="DEATH_EVENT", palette=cols)
Days_of_week.set_title("Distribution Of Age", color="#774571")
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Define non-binary features
features = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time"]
cols = ["#32CD32", "#FF0000"]  # Palet warna untuk plot

# Calculate descriptive statistics
desc_stats = df[features].describe().T

# Detect outliers using IQR
outliers = {}
for feature in features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers[feature] = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]

# Save descriptive statistics to CSV
desc_stats.to_csv('descriptive_statistics.csv')
print("Descriptive statistics have been saved to descriptive_statistics.csv")

# Save outlier information to CSV
outliers_combined = pd.concat(outliers, axis=0)
outliers_combined.to_csv('outliers.csv')
print("Outlier information has been saved to outliers.csv")

# Plotting
for feature in features: 
    plt.figure(figsize=(10, 7))
    sns.swarmplot(x=df["DEATH_EVENT"], y=df[feature], color="black", alpha=0.7)
    sns.boxenplot(x=df["DEATH_EVENT"], y=df[feature], palette=cols)
    plt.title(f'Boxen and Swarm Plot for {feature}')
    plt.xlabel('DEATH_EVENT')
    plt.ylabel(feature)
    plt.show()

# Plotting "Kernel Density Estimation (kde plot)" of time and age features 
plt.figure(figsize=(10, 7))
sns.kdeplot(x=df["time"], y=df["age"], hue=df["DEATH_EVENT"], palette=cols)
plt.title('Kernel Density Estimation of Time and Age by Death Event')
plt.xlabel('Time')
plt.ylabel('Age')
plt.show()

# DATA PREPROCESSING
# Defining independent and dependent attributes in training and test sets
X=df.drop(["DEATH_EVENT"],axis=1)
y=df["DEATH_EVENT"]

# Setting up a standard scaler for the features and analyzing it thereafter
# Mengambil nama kolom dari DataFrame X
col_names = list(X.columns)

# Membuat objek StandardScaler dari sklearn
s_scaler = preprocessing.StandardScaler()

# Melakukan fit dan transformasi pada data X, sehingga data tersebut di-skalakan
X_scaled = s_scaler.fit_transform(X)

# Mengubah array hasil scaling menjadi DataFrame dan menetapkan nama kolom sesuai dengan nama kolom asli
X_scaled = pd.DataFrame(X_scaled, columns=col_names)

# Menyimpan DataFrame hasil scaling ke dalam file CSV
X_scaled.to_csv('scaled_features.csv', index=False)
print("Hasil scaling telah disimpan ke scaled_features.csv")

# Menampilkan deskripsi statistik dari DataFrame hasil scaling
print(X_scaled.describe().T)

#Plotting the scaled features using boxen plots
colors =["#CD5C5C","#F08080","#FA8072","#E9967A","#FFA07A"]
plt.figure(figsize=(20,10))
sns.boxenplot(data = X_scaled,palette = colors)
plt.xticks(rotation=60)
plt.show()

#spliting variables into training and test sets
X_train, X_test, y_train,y_test = train_test_split(X_scaled,y,test_size=0.30,random_state=25)

#REGRESSION LOGISTIC
# Definisikan semua kolom independen
featureslr = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
            "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium",
            "sex", "smoking", "time"]

# Memisahkan variabel independen (X) dan dependen (y)
X = df[featureslr]
y = df['DEATH_EVENT']

# Menambahkan konstanta untuk model OLS
X = sm.add_constant(X)

# Membuat model regresi logistik
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Menampilkan ringkasan model
print(result.summary())

# Mengambil koefisien, p-value, dan pseudo R-squared
coefficients = result.params
p_values = result.pvalues
pseudo_r_squared = result.prsquared

# Menyimpan hasil ke CSV
summary_df = pd.DataFrame({'Coefficient': coefficients, 'P-Value': p_values})
summary_df.loc['Pseudo R-squared'] = [pseudo_r_squared, '']
summary_df.to_csv('logistic_regression_summary.csv', index=True)
print("Ringkasan hasil regresi logistik telah disimpan ke logistic_regression_summary.csv")



# MODEL BUILDING
# WITH SUPPORT VECTOR MACHINE / SVM MODEL
# SVM Model
# Instantiating the SVM algorithm 
model_svm = svm.SVC()

# Fitting the model 
model_svm.fit(X_train, y_train)

# Predicting the test variables
y_pred_svm = model_svm.predict(X_test)

# Getting the score 
score_svm = model_svm.score(X_test, y_test)
print(f'Score: {score_svm}')

# Printing classification report (since there was biasness in target labels)
class_report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
print(classification_report(y_test, y_pred_svm))

# Converting classification report to DataFrame and saving to CSV
class_report_svm_df = pd.DataFrame(class_report_svm).transpose()
class_report_svm_df.to_csv('classification_report_svm.csv', index=True)
print("Classification report telah disimpan ke classification_report_svm.csv")

# Getting the confusion matrix
cf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# Normalizing confusion matrix
cf_matrix_svm_normalized = cf_matrix_svm / np.sum(cf_matrix_svm)

# Converting confusion matrix to DataFrame and saving to CSV
cf_matrix_svm_df = pd.DataFrame(cf_matrix_svm_normalized, index=model_svm.classes_, columns=model_svm.classes_)
cf_matrix_svm_df.to_csv('confusion_matrix_svm.csv', index=True)
print("Confusion matrix telah disimpan ke confusion_matrix_svm.csv")

# Plotting the confusion matrix
cmap1 = sns.diverging_palette(2, 165, s=80, l=55, n=9)
plt.subplots(figsize=(10,7))
sns.heatmap(cf_matrix_svm_normalized, cmap=cmap1, annot=True, annot_kws={'size':25}, fmt=".2f")
plt.title('Confusion Matrix SVM')
plt.show()


#ANN
# Load dataset
data = pd.read_csv("heart_failure.csv")

# Features and target
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the labels
y_train_onehot = np.zeros((y_train.size, y_train.max()+1))
y_train_onehot[np.arange(y_train.size), y_train] = 1

# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

class SimpleANN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

        # Choose activation function
        if activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative

    def forward(self, X):
        # Forward pass
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = self.activation(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.output = sigmoid(self.output_input)  # Output layer always uses sigmoid

        return self.output

    def backward(self, X, y, learning_rate):
        # Backward pass
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output_input)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_input)

        # Update weights
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.weights_input_hidden += learning_rate * np.dot(X.T, hidden_delta)

    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

# Train the model with ReLU activation
ann_relu = SimpleANN(input_size=X_train.shape[1], hidden_size=10, output_size=2, activation='relu')
ann_relu.train(X_train, y_train_onehot, epochs=1000, learning_rate=0.01)

# Train the model with Sigmoid activation
ann_sigmoid = SimpleANN(input_size=X_train.shape[1], hidden_size=10, output_size=2, activation='sigmoid')
ann_sigmoid.train(X_train, y_train_onehot, epochs=1000, learning_rate=0.01)

# Train the model with Tanh activation
ann_tanh = SimpleANN(input_size=X_train.shape[1], hidden_size=10, output_size=2, activation='tanh')
ann_tanh.train(X_train, y_train_onehot, epochs=1000, learning_rate=0.01)

# Predict with the ReLU model
y_pred_ann_relu = np.argmax(ann_relu.forward(X_test), axis=1)
accuracy_ann_relu = accuracy_score(y_test, y_pred_ann_relu)
print("Classification report for ANN (ReLU):")
print(classification_report(y_test, y_pred_ann_relu))
print(f"Accuracy for ANN (ReLU): {accuracy_ann_relu:.4f}")

# Predict with the Sigmoid model
y_pred_ann_sigmoid = np.argmax(ann_sigmoid.forward(X_test), axis=1)
accuracy_ann_sigmoid = accuracy_score(y_test, y_pred_ann_sigmoid)
print("Classification report for ANN (Sigmoid):")
print(classification_report(y_test, y_pred_ann_sigmoid))
print(f"Accuracy for ANN (Sigmoid): {accuracy_ann_sigmoid:.4f}")

# Predict with the Tanh model
y_pred_ann_tanh = np.argmax(ann_tanh.forward(X_test), axis=1)
accuracy_ann_tanh = accuracy_score(y_test, y_pred_ann_tanh)
print("Classification report for ANN (Tanh):")
print(classification_report(y_test, y_pred_ann_tanh))
print(f"Accuracy for ANN (Tanh): {accuracy_ann_tanh:.4f}")

# Save the classification report for the ReLU model
class_report_ann_relu = classification_report(y_test, y_pred_ann_relu, output_dict=True)
class_report_ann_relu_df = pd.DataFrame(class_report_ann_relu).transpose()
class_report_ann_relu_df.to_csv('classification_report_ann_relu.csv', index=True)
print("Classification report (ReLU) has been saved to classification_report_ann_relu.csv")

# Save the classification report for the Sigmoid model
class_report_ann_sigmoid = classification_report(y_test, y_pred_ann_sigmoid, output_dict=True)
class_report_ann_sigmoid_df = pd.DataFrame(class_report_ann_sigmoid).transpose()
class_report_ann_sigmoid_df.to_csv('classification_report_ann_sigmoid.csv', index=True)
print("Classification report (Sigmoid) has been saved to classification_report_ann_sigmoid.csv")

# Save the classification report for the Tanh model
class_report_ann_tanh = classification_report(y_test, y_pred_ann_tanh, output_dict=True)
class_report_ann_tanh_df = pd.DataFrame(class_report_ann_tanh).transpose()
class_report_ann_tanh_df.to_csv('classification_report_ann_tanh.csv', index=True)
print("Classification report (Tanh) has been saved to classification_report_ann_tanh.csv")

# Visualisasi Classification Report (ReLU)
plt.figure(figsize=(8, 6))
sns.heatmap(class_report_ann_relu_df.drop(columns=['support']).T, annot=True, cmap='Blues')
plt.title('Classification Report (ReLU)')
plt.show()

# Visualisasi Classification Report (Sigmoid)
plt.figure(figsize=(8, 6))
sns.heatmap(class_report_ann_sigmoid_df.drop(columns=['support']).T, annot=True, cmap='Blues')
plt.title('Classification Report (Sigmoid)')
plt.show()

# Visualisasi Classification Report (Tanh)
plt.figure(figsize=(8, 6))
sns.heatmap(class_report_ann_tanh_df.drop(columns=['support']).T, annot=True, cmap='Blues')
plt.title('Classification Report (Tanh)')
plt.show()






