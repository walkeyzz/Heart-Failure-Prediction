# HEART FAILURE PREDICTION EXPLANATION CODE
by Keiko joceliandita 4ITE1

# Import Libraries
```
import warnings
warnings.filterwarnings('ignore')
```
Untuk menonaktifkan semua pesan peringatan (warnings). Ini supaya output nya bersih, minus nya bisa menyembunyikan peringatan penting

```
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Mengatur level logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Menonaktifkan operasi custom oneDNN
```
Bersumber dari GPT, baris-baris ini mengonfigurasi lingkungan untuk TensorFlow. TF_CPP_MIN_LOG_LEVEL diatur ke '2' untuk menampilkan hanya pesan error dan mengabaikan pesan info dan warning. 
TF_ENABLE_ONEDNN_OPTS diatur ke '0' untuk menonaktifkan operasi custom oneDNN yang tidak kompatibel di environ laptop saya 

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import tensorflow as tf
```
1. numpy: Digunakan untuk operasi numerik, terutama array dan matriks.
2. pandas: Digunakan untuk manipulasi dan analisis data berbasis tabel (DataFrame).
3. matplotlib.pyplot: Digunakan untuk membuat visualisasi data, seperti grafik dan plot.
4. seaborn: Dibangun di atas Matplotlib, digunakan untuk membuat visualisasi statistik yang lebih atraktif dan informatif.
5. statsmodels.api: Digunakan untuk analisis statistik dan ekonometrik.
6. tensorflow: Digunakan untuk machine learning dan deep learning.

# Mengimpor Library Scikit-Learn
```
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
```
1. sklearn: Merupakan library untuk machine learning di Python yang menyediakan alat-alat untuk pemrosesan data, algoritma pembelajaran, dan evaluasi model.
2. preprocessing: Berisi alat-alat untuk pra-pemrosesan data, seperti normalisasi dan encoding.
3. StandardScaler: Digunakan untuk menstandarkan fitur dengan menghapus rata-rata dan menskalakan ke varian satuan.
4. train_test_split: Digunakan untuk membagi dataset menjadi set pelatihan dan pengujian.
5. svm: Berisi algoritma Support Vector Machine untuk klasifikasi dan regresi.
6. classification_report, accuracy_score, precision_score, recall_score, confusion_matrix, f1_score: Fungsi-fungsi ini digunakan untuk mengevaluasi kinerja model klasifikasi dengan berbagai metrik.

# Loading Data
```
df = pd.read_csv(r"heart_failure.csv")
print(df.head())
```
![WhatsApp Image 2024-06-12 at 18 10 07_3b16a81a](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/2c6c9f6e-fdce-4f9d-9343-66a34eac459d)
```
print(df.info())
```
![WhatsApp Image 2024-06-12 at 18 10 52_c927dac7](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/358f4c91-c4f7-44a3-9003-f06abbc93ccd)

# Menghitung Skewness
Skewness adalah kemiringan data, ini penting karena jika adanya skewness maka distribusi data tidak merata dan mempengaruhi ke hasil analisis nanti, jika ini terjadi maka harus melakukan transformasi data terlebih dahulu
```
skewness = df["DEATH_EVENT"].skew()
print(f'Skewness of DEATH_EVENT: {skewness}')
```
Menghitung skewness dari kolom DEATH_EVENT
![WhatsApp Image 2024-06-12 at 18 24 19_66295698](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/e49a82c5-9613-4e82-8ce6-e42476e149a0)
Hasil skewness >0 maka data miring ke kanan, <0 maka data miring ke kiri, hasil 0.77 masih bisa dikatakan normal sehingga kita tidak perlu mengubah/mentransformasi data nya

# Menyimpan Nilai Skewness
```
skewness_info = pd.DataFrame({
    'Feature': ['DEATH_EVENT'],
    'Skewness': [skewness]
})
```
```
skewness_info.to_csv('skewness_info.csv', index=False)
print("Informasi skewness telah diekspor ke skewness_info.csv")
```
Ini metode dari pandas untuk menyimpan DataFrame sebagai file CSV. index=False berarti indeks tidak akan disertakan dalam file CSV. 

# Visualisasi Skewness dengan Python
```
plt.figure(figsize=(8, 6))
cols= ["#32CD32","#FF0000"]
ax = sns.countplot(x=df["DEATH_EVENT"], palette=cols)
ax.bar_label(ax.containers[0])
plt.title('Count Plot of DEATH_EVENT')
plt.xlabel('DEATH_EVENT')
plt.ylabel('Count')
plt.show()
```
Disini kita menggunakan fungsi dari seaborn untuk membuat count plot dan Fungsi dari matplotlib untuk mengatur judul, label dan show plot
![WhatsApp Image 2024-05-22 at 23 20 58_98186414](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/083bd94d-b088-4015-8854-77497fad17ff)

# Visualisasi Skewness dengan PowerBI
![WhatsApp Image 2024-06-12 at 18 46 02_45e87e6a](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/a60c9809-edcd-4c9b-9164-e49d59b0364f)

# Deskripsi Statistik
```
description = df.describe().T
print(description)
```
fungsi dari pandas yang digunakan untuk menghitung statistik deskriptif dari dataset. Statistik yang dihitung meliputi count (jumlah data), mean (rata-rata), std (standar deviasi), min (nilai minimum), 25% (kuartil pertama), 50% (median atau kuatril kedua), 75% (kuartil ketiga), dan max (nilai maksimum). Hasilnya berupa DataFrame yang berisi statistik tersebut

![WhatsApp Image 2024-06-12 at 18 56 17_ca52b4ec](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/79227db6-781e-4537-8d93-cf389f46f1c7)

"75%", itu mengindikasikan nilai di mana 75% data berada di bawah atau sama dengan nilai tersebut

# Analisis Bivariat dengan matriks korelasi menggunakan heatmap
Untuk lihat hubungan keseluruhan data independen dan dependen nya
```
cmap = sns.diverging_palette(2, 165, s=80, l=55, n=9) #Pengaturan warna untuk heatmap dari library seaborn
corrmat = df.corr()  # Hitung matriks korelasi dari .corr pandas
plt.subplots(figsize=(20, 20))
sns.heatmap(corrmat, cmap=cmap, annot=True, square=True)
plt.title('Heatmap of Correlation Matrix')
plt.show()
```

# Simpan Correlation Matrix to file CSV
```
corrmat.to_csv('correlation_matrix.csv')
print("Matriks korelasi telah diekspor ke correlation_matrix.csv")
```

# Hasil visualisasi / Heatmap dengan Python
![WhatsApp Image 2024-05-22 at 23 21 21_5c272075](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/a87a41b4-e203-45c6-a719-e655499f7398)

# Hasil Visualisasi dengan PowerBI
![WhatsApp Image 2024-06-12 at 22 38 35_2d494393](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/9021d034-59f4-4ed6-9933-0ea62dfe0847)
1. "Time" itu fitur yang paling penting karena kalau bisa didiagnosa masalah kardiovaskularnya lebih awal, bisa dapet pengobatan tepat waktu dan ngurangin risiko kematian. Ini keliatan dari hubungan terbaliknya.
2. "Serum_creatinine" juga penting karena serum (komponen penting dalam darah) yang banyak di darah bikin kerja jantung jadi lebih gampang.
3. "Ejection_fraction" juga punya pengaruh besar ke variabel target, ini wajar karena ejection fraction itu pada dasarnya efisiensi kerja jantung.
4. Bisa diliat dari pola hubungan terbaliknya kalau fungsi jantung makin menurun seiring bertambahnya usia.

# Membuat DataFrame untuk data yang dihitung
```
age_death_count = df.groupby(['age', 'DEATH_EVENT']).size().unstack(fill_value=0)
```

# Simpan DataFrame ke CSV
```
age_death_count.to_csv('age_death_distribution.csv')
print("Data distribusi usia berdasarkan kejadian kematian telah diekspor ke age_death_distribution.csv")
```

# Visualisasi distribusi usia berdasarkan kejadian kematian
```
plt.figure(figsize=(15, 10))
cols = ["#32CD32", "#FF0000"]  # Palet warna untuk batang
Days_of_week = sns.countplot(x=df['age'], data=df, hue="DEATH_EVENT", palette=cols)
Days_of_week.set_title("Distribution Of Age", color="#774571")
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
```

# Hasil Visualisasi dengan Python
![WhatsApp Image 2024-05-23 at 00 23 22_bf7dd96f](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/7a5e2ffe-ad15-4460-8c99-691a5599d007)

# Hasil Visualisasi dengan PowerBI
![WhatsApp Image 2024-06-12 at 23 12 05_e32029fd](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/93cae964-dd39-4a7e-ac71-d34a030d70dd)
Disini kita ingin menghitung distribusi usia berdasarkan kejadian kematian dalam dataset

# Define non-binary features
```
features = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time"]
cols = ["#32CD32", "#FF0000"]  # Palet warna untuk plot
```
Mendefinisikan fitur-fitur non-biner yang akan dianalisis.

# Calculate descriptive statistics
```
desc_stats = df[features].describe().T
```
Menghitung statistik deskriptif (misalnya, mean, median, quartiles) untuk fitur-fitur yang didefinisikan dan mentransposisi hasilnya.

# Detect outliers using IQR
Outlier adalah nilai-nilai yang berbeda secara signifikan dari sebagian besar data. Deteksi outlier penting untuk memastikan analisis data kita akurat dan tidak dipengaruhi oleh data yang ekstrem atau tidak biasa. Salah satu metode untuk mendeteksi outlier adalah dengan menggunakan Interquartile Range (IQR).
```
outliers = {}
for feature in features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers[feature] = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
```
1. Hitung Q1 & Q3
2. Hitung IQR dengan nilai Q3-Q1
3. Batas bawah dihitung sebagai Q1 dikurangi 1.5 kali IQR.
4. Batas atas dihitung sebagai Q3 ditambah 1.5 kali IQR.
5. Nilai-nilai di luar batas ini dianggap sebagai outlier.
6. Kita memfilter data untuk setiap fitur yang nilainya lebih kecil dari batas bawah atau lebih besar dari batas atas.
7. Data yang memenuhi kondisi tersebut disimpan dalam dictionary outliers dengan nama fitur sebagai kunci.
8. Looping untuk setiap features 

# Save descriptive statistics to CSV
```
desc_stats.to_csv('descriptive_statistics.csv')
print("Descriptive statistics have been saved to descriptive_statistics.csv")
```

# Save outlier information to CSV
```
outliers_combined = pd.concat(outliers, axis=0)
outliers_combined.to_csv('outliers.csv')
print("Outlier information has been saved to outliers.csv")
```
Menggabungkan semua informasi outlier menjadi satu DataFrame dan disimpan dalam file outliers.csv

# Plotting Outlier
```
for feature in features: 
    plt.figure(figsize=(10, 7))
    sns.swarmplot(x=df["DEATH_EVENT"], y=df[feature], color="black", alpha=0.7)
    sns.boxenplot(x=df["DEATH_EVENT"], y=df[feature], palette=cols)
    plt.title(f'Boxen and Swarm Plot for {feature}')
    plt.xlabel('DEATH_EVENT')
    plt.ylabel(feature)
    plt.show()
```
# Hasil Visualisasi Menggunakan Python
![WhatsApp Image 2024-05-23 at 09 36 50_ad862f37](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/36eeda83-fd09-4672-a316-3ace919c0862)
![WhatsApp Image 2024-05-23 at 09 37 23_8f1c8fcc](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/2f720db2-7571-4821-83e8-dbb79adccafc)
![WhatsApp Image 2024-05-23 at 09 37 45_5704a2a2](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/f956eb36-019e-4e49-8e68-82bde57436dc)
![WhatsApp Image 2024-05-23 at 09 38 08_a0d9629f](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/26b8e742-82fc-4d3d-94e9-a1f7f882daf5)
![WhatsApp Image 2024-05-23 at 09 38 45_86104c1a](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/7b82ec2e-0e81-4132-a97e-e1bc3be82924)
![WhatsApp Image 2024-05-23 at 09 39 14_99be0fe0](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/0ac2d661-871d-4213-859f-caca5fdfba80)
![WhatsApp Image 2024-05-23 at 09 39 41_12a8ed3f](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/c4404b19-56f0-482b-97cd-32dee4a05c0f)

# Hasil Visualisasi Menggunakan Python Visualization PowerBI
![WhatsApp Image 2024-06-13 at 00 22 42_30eee1eb](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/e0d0b630-d747-4c08-bf46-8c4c3477d62c)
![WhatsApp Image 2024-06-13 at 00 21 25_dfbd73a2](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/367e436c-37f6-45be-84cd-e6ad5d3b584d)
![WhatsApp Image 2024-06-13 at 00 23 55_a3f5f2ab](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/e96b6d9a-6a37-4aac-82d9-a21a7a74140a)
![WhatsApp Image 2024-06-13 at 00 30 21_e4d6ec14](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/38f0ff31-effd-4274-b224-33b027e5a74e)
![WhatsApp Image 2024-06-13 at 00 32 26_7fc16af2](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/6b321a5d-f5bd-4038-834e-397c9da87da0)
![WhatsApp Image 2024-06-13 at 00 33 37_31802cee](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/462fe803-2c65-4661-bd0a-d6dc3123a074)
![WhatsApp Image 2024-06-13 at 00 34 41_860a8817](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/f0f8bd70-199c-471a-a372-2287e2312cda)

Kita melihat beberapa data yang di luar kebiasaan (outliers) pada hampir semua fitur. Namun, mengingat ukuran dataset dan relevansinya, kita tidak akan menghapus outliers semacam itu dalam tahap preprocessing data karena hal tersebut tidak akan membawa dampak statistik yang signifikan.

# Plotting "Kernel Density Estimation (kde plot)" of time and age features 
```
plt.figure(figsize=(10, 7))
sns.kdeplot(x=df["time"], y=df["age"], hue=df["DEATH_EVENT"], palette=cols) #kde is kernel density estimation (KDE) 
plt.title('Kernel Density Estimation of Time and Age by Death Event')
plt.xlabel('Time')
plt.ylabel('Age')
plt.show()
```

# Hasil Visualisasi dengan Python
![WhatsApp Image 2024-05-23 at 09 46 33_ff7268d1](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/d9b3e91d-e8ca-4a56-abd0-7c4f94d3bf86)

# Hasil Visualisasi Menggunakan Python Visualization PowerBI
![image](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/0842a510-07cf-405f-ac0a-54288efbeaef)

Dengan jumlah hari follow-up yang lebih sedikit, pasien seringkali meninggal hanya ketika usia mereka semakin tua. Semakin lama jumlah hari follow-up, semakin besar kemungkinannya, bahwa terjadi kejadian fatalitas.

# DATA PREPROCESSING
# Defining independent and dependent attributes in training and test sets
```
X=df.drop(["DEATH_EVENT"],axis=1) #Menghapus kolom DEATH_EVENT
y=df["DEATH_EVENT"]
```

# Setting up a standard scaler for the features and analyzing it thereafter
# Mengambil nama kolom dari DataFrame X
```
col_names = list(X.columns)
```
# Membuat objek StandardScaler dari sklearn
StandardScaler adalah function dari sklearn. Guna nya untuk standarisasi fitur variabel X, agar nilai nya gak besar besar amat. Fungsi nya dalam Machine Learning ialah :
1. Mempermudah Interpretasi Model
2. Menghindari Dominasi Fitur, sehingga model dapat fokus pada semua fitur dengan proporsi yang seimbang
3. Meningkatkan Konvergensi Algoritma, Standardisasi membantu algoritma konvergen lebih cepat dan dengan hasil yang lebih baik.

Proses nya secara Matematis 
```
Scaled Value = (Original Value - Mean) / Standard Deviation
```
```
s_scaler = preprocessing.StandardScaler()
```

# Melakukan fit dan transformasi pada data X, sehingga data tersebut di-skalakan
X_scaled = s_scaler.fit_transform(X)

# Mengubah array hasil scaling menjadi DataFrame dan menetapkan nama kolom sesuai dengan nama kolom asli
X_scaled = pd.DataFrame(X_scaled, columns=col_names)

# Menyimpan DataFrame hasil scaling ke dalam file CSV
X_scaled.to_csv('scaled_features.csv', index=False)
print("Hasil scaling telah disimpan ke scaled_features.csv")

# Menampilkan deskripsi statistik dari DataFrame hasil scaling
print(X_scaled.describe().T)

# Hasil Output nya
![image](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/725e2c3f-6bec-4beb-8dde-1de6351f73d1)

#Plotting the scaled features using boxen plots
colors =["#CD5C5C","#F08080","#FA8072","#E9967A","#FFA07A"]
plt.figure(figsize=(20,10))
sns.boxenplot(data = X_scaled,palette = colors)
plt.xticks(rotation=60)
plt.show()

# Hasil Visualisasi dengan Python
![WhatsApp Image 2024-05-25 at 16 36 07_36a3ad88](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/6e58ffdf-86ba-4d01-be68-bfc142165c48)

# Hasil Visualisasi Menggunakan Python Visualization PowerBI
![image](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/90329f74-b143-4b62-977e-f88a4cc3998c)

#spliting variables into training and test sets
X_train, X_test, y_train,y_test = train_test_split(X_scaled,y,test_size=0.30,random_state=25)

# REGRESSION LOGISTIC
Membangun model prediktif untuk data Biner ( 0 dan 1)
# Definisikan semua kolom independen
```
featureslr = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
            "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium",
            "sex", "smoking", "time"]
```

# Memisahkan variabel independen (X) dan dependen (y)
```
X = df[featureslr]
y = df['DEATH_EVENT']
```

# Menambahkan konstanta untuk model OLS
```
X = sm.add_constant(X) #menggunakan fungsi add_constant dari statsmodels untuk model regresi logistik.
```

# Membuat model regresi logistik
```
logit_model = sm.Logit(y, X)
result = logit_model.fit()
```
Membuat model regresi logistik dengan menggunakan Logit dari statsmodels dan menghitung hasilnya dengan memanggil metode fit()
# Menampilkan ringkasan model
```
print(result.summary())
```
![image](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/f6f697d5-75ab-4256-a1ac-f05fb5670c03)

# Mengambil koefisien, p-value, dan pseudo R-squared
```
coefficients = result.params
p_values = result.pvalues
pseudo_r_squared = result.prsquared
```
Nilai koefisien, p-value, dan pseudo R-squared diambil dari hasil model untuk analisis lebih lanjut.

# Menyimpan hasil ke CSV
```
summary_df = pd.DataFrame({'Coefficient': coefficients, 'P-Value': p_values})
summary_df.loc['Pseudo R-squared'] = [pseudo_r_squared, '']
summary_df.to_csv('logistic_regression_summary.csv', index=True)
print("Ringkasan hasil regresi logistik telah disimpan ke logistic_regression_summary.csv")
```

# MODEL BUILDING
# WITH SUPPORT VECTOR MACHINE / SVM MODEL
# SVM Model
# Instantiating the SVM algorithm 
```
model_svm = svm.SVC()
```

# Fitting the model 
```
model_svm.fit(X_train, y_train)
```

# Predicting the test variables
```
y_pred_svm = model_svm.predict(X_test)
```

# Getting the score 
```
score_svm = model_svm.score(X_test, y_test)
print(f'Score: {score_svm}')
```
Kemudian, kita melakukan prediksi terhadap variabel test menggunakan model_svm.predict(X_test) dan menghitung skor akurasi model menggunakan model_svm.score(X_test, y_test) yang akan dicetak ke layar.
![image](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/a58d7514-2a38-457a-8410-363e21dc7833)
Model SVM yang telah dilatih memiliki akurasi sekitar 78.89% pada data uji (test data).

# Printing classification report 
```
class_report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
print(classification_report(y_test, y_pred_svm))
```
![image](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/7b69087c-83b0-4bc7-bc79-f4f76564177a)
1. Precision: Mengukur ketepatan prediksi positif.
2. Recall: Mengukur kemampuan model dalam menemukan semua sampel positif.
3. F1-score: Harmonic mean dari precision dan recall, memberikan gambaran yang lebih seimbang tentang kinerja model.
4. Support: Jumlah sampel sebenarnya dalam setiap kelas.

# Converting classification report to DataFrame and saving to CSV
```
class_report_svm_df = pd.DataFrame(class_report_svm).transpose()
class_report_svm_df.to_csv('classification_report_svm.csv', index=True)
print("Classification report telah disimpan ke classification_report_svm.csv")
```

# Getting the confusion matrix
```
cf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
```
# Normalizing confusion matrix
```
cf_matrix_svm_normalized = cf_matrix_svm / np.sum(cf_matrix_svm)
```
Selanjutnya, kita menghitung confusion matrix untuk evaluasi performa model. Confusion matrix ini juga dinormalisasi untuk menghasilkan nilai antara 0 dan 1.
# Converting confusion matrix to DataFrame and saving to CSV
```
cf_matrix_svm_df = pd.DataFrame(cf_matrix_svm_normalized, index=model_svm.classes_, columns=model_svm.classes_)
cf_matrix_svm_df.to_csv('confusion_matrix_svm.csv', index=True)
print("Confusion matrix telah disimpan ke confusion_matrix_svm.csv")
```
Terakhir, kita melakukan plotting confusion matrix yang telah dinormalisasi menggunakan seaborn heatmap untuk visualisasi evaluasi performa model SVM. 

# Plotting the confusion matrix
```
cmap1 = sns.diverging_palette(2, 165, s=80, l=55, n=9)
plt.subplots(figsize=(10,7))
sns.heatmap(cf_matrix_svm_normalized, cmap=cmap1, annot=True, annot_kws={'size':25}, fmt=".2f")
plt.title('Confusion Matrix SVM')
plt.show()
```
Terakhir, kita melakukan plotting confusion matrix yang telah dinormalisasi menggunakan seaborn heatmap untuk visualisasi evaluasi performa model SVM. 

# Hasil Visualisasi Menggunakan Python 
![WhatsApp Image 2024-05-25 at 16 57 58_6a4633c5](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/de955d97-a99e-4ee7-9bfe-cd18d2e97be4)

# Hasil Visualisasi Menggunakan Python Visualization PowerBI
![image](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/879821cf-9e01-48f0-86e8-e4c67084ed1b)

# ANN
# Load dataset
```
data = pd.read_csv("heart_failure.csv")
```

# Features and target
```
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']
```

# Split the data into training and testing sets
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

# Standardize the features
```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
# One-hot encode the labels
Fungsi aktivasi pada output layer seperti softmax tidak akan berfungsi dengan benar tanpa one-hot encoding. Ini bisa menyebabkan kesulitan dalam konvergensi model dan perhitungan error yang tidak akurat.
```
y_train_onehot = np.zeros((y_train.size, y_train.max()+1))
y_train_onehot[np.arange(y_train.size), y_train] = 1
```
# Activation functions and their derivatives
```
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
# Fungsi aktivasi (ReLU, sigmoid, tanh) dan turunan mereka untuk digunakan dalam ANN.

class SimpleANN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.input_size = input_size #Jumlah Neuron Input
        self.hidden_size = hidden_size #Jumlah neuron di lapisan tersembunyi.
        self.output_size = output_size #Jumlah Neuron Output
# Inisialisasi parameter untuk ukuran input, hidden layer, dan output dari jaringan saraf.

        # Menginisialisasi bobot (weights) antara lapisan input dan lapisan tersembunyi, serta antara lapisan tersembunyi dan lapisan output dengan nilai acak.
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
        # Forward pass, Melakukan forward pass dari jaringan saraf untuk menghitung keluaran (output) berdasarkan input X.
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = self.activation(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.output = sigmoid(self.output_input)  # Output layer always uses sigmoid

        return self.output

    def backward(self, X, y, learning_rate):
        # Backward pass, Melakukan backward pass untuk menghitung gradien dan memperbarui bobot berdasarkan kesalahan (error) antara output prediksi dan nilai sebenarnya y.
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output_input)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_input)

        # Update weights
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.weights_input_hidden += learning_rate * np.dot(X.T, hidden_delta)

        # Melatih jaringan saraf dengan melakukan forward dan backward pass selama sejumlah epochs.
    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
```

# Train the model with ReLU activation
```
ann_relu = SimpleANN(input_size=X_train.shape[1], hidden_size=10, output_size=2, activation='relu')
ann_relu.train(X_train, y_train_onehot, epochs=1000, learning_rate=0.01)
```

# Train the model with Sigmoid activation
```
ann_sigmoid = SimpleANN(input_size=X_train.shape[1], hidden_size=10, output_size=2, activation='sigmoid')
ann_sigmoid.train(X_train, y_train_onehot, epochs=1000, learning_rate=0.01)
```

# Train the model with Tanh activation
```
ann_tanh = SimpleANN(input_size=X_train.shape[1], hidden_size=10, output_size=2, activation='tanh')
ann_tanh.train(X_train, y_train_onehot, epochs=1000, learning_rate=0.01)
```

# Predict with the ReLU model
```
y_pred_ann_relu = np.argmax(ann_relu.forward(X_test), axis=1)
accuracy_ann_relu = accuracy_score(y_test, y_pred_ann_relu)
print("Classification report for ANN (ReLU):")
print(classification_report(y_test, y_pred_ann_relu))
print(f"Accuracy for ANN (ReLU): {accuracy_ann_relu:.4f}")
```
1. y_pred_ann_relu = np.argmax(ann_relu.forward(X_test), axis=1): Menggunakan model ANN dengan aktivasi ReLU (ann_relu) untuk melakukan prediksi pada data uji (X_test).
2. np.argmax digunakan untuk mendapatkan indeks nilai maksimum dari setiap baris, yang mengindikasikan kelas prediksi.
3. accuracy_ann_relu = accuracy_score(y_test, y_pred_ann_relu): Menghitung akurasi model ANN dengan aktivasi ReLU berdasarkan prediksi (y_pred_ann_relu) dan label sebenarnya dari data uji (y_test).
# Predict with the Sigmoid model
```
y_pred_ann_sigmoid = np.argmax(ann_sigmoid.forward(X_test), axis=1)
accuracy_ann_sigmoid = accuracy_score(y_test, y_pred_ann_sigmoid)
print("Classification report for ANN (Sigmoid):")
print(classification_report(y_test, y_pred_ann_sigmoid))
print(f"Accuracy for ANN (Sigmoid): {accuracy_ann_sigmoid:.4f}")
```

# Predict with the Tanh model
```
y_pred_ann_tanh = np.argmax(ann_tanh.forward(X_test), axis=1)
accuracy_ann_tanh = accuracy_score(y_test, y_pred_ann_tanh)
print("Classification report for ANN (Tanh):")
print(classification_report(y_test, y_pred_ann_tanh))
print(f"Accuracy for ANN (Tanh): {accuracy_ann_tanh:.4f}")
```

# Save the classification report for the ReLU model
```
class_report_ann_relu = classification_report(y_test, y_pred_ann_relu, output_dict=True)
class_report_ann_relu_df = pd.DataFrame(class_report_ann_relu).transpose()
class_report_ann_relu_df.to_csv('classification_report_ann_relu.csv', index=True)
print("Classification report (ReLU) has been saved to classification_report_ann_relu.csv")
```
# Save the classification report for the Sigmoid model
```
class_report_ann_sigmoid = classification_report(y_test, y_pred_ann_sigmoid, output_dict=True)
class_report_ann_sigmoid_df = pd.DataFrame(class_report_ann_sigmoid).transpose()
class_report_ann_sigmoid_df.to_csv('classification_report_ann_sigmoid.csv', index=True)
print("Classification report (Sigmoid) has been saved to classification_report_ann_sigmoid.csv")
```
# Save the classification report for the Tanh model
```
class_report_ann_tanh = classification_report(y_test, y_pred_ann_tanh, output_dict=True)
class_report_ann_tanh_df = pd.DataFrame(class_report_ann_tanh).transpose()
class_report_ann_tanh_df.to_csv('classification_report_ann_tanh.csv', index=True)
print("Classification report (Tanh) has been saved to classification_report_ann_tanh.csv")
```
# Visualisasi Classification Report (ReLU)
```
plt.figure(figsize=(8, 6))
sns.heatmap(class_report_ann_relu_df.drop(columns=['support']).T, annot=True, cmap='Blues')
plt.title('Classification Report (ReLU)')
plt.show()
```
![image](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/361e86ea-7fef-4441-bcee-86419be4b438)
![image](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/98263171-9119-475b-98e8-51d30ca5462b)


# Visualisasi Classification Report (Sigmoid)
```
plt.figure(figsize=(8, 6))
sns.heatmap(class_report_ann_sigmoid_df.drop(columns=['support']).T, annot=True, cmap='Blues')
plt.title('Classification Report (Sigmoid)')
plt.show()
```
![image](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/bbbc053a-1b9a-45bc-b6d0-162e9bcf401c)

![image](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/f42bf1ee-f3f3-41f2-a38a-b259075f04a3)


# Visualisasi Classification Report (Tanh)
```
plt.figure(figsize=(8, 6))
sns.heatmap(class_report_ann_tanh_df.drop(columns=['support']).T, annot=True, cmap='Blues')
plt.title('Classification Report (Tanh)')
plt.show()
```
![image](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/b6254965-f64f-40df-b243-2d7c897136c0)

![image](https://github.com/walkeyzz/Heart-Failure-Prediction/assets/146419451/833daaea-c207-478c-a475-47c471ebff0f)











