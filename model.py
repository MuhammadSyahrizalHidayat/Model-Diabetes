import numpy as np  # Mengimpor library NumPy untuk manipulasi data numerik.
import pandas as pd  # Mengimpor library Pandas untuk menangani data dalam format DataFrame.
import pickle  # Mengimpor library Pickle untuk menyimpan dan memuat model yang telah dilatih.
from sklearn.model_selection import train_test_split  # Mengimpor fungsi train_test_split dari scikit-learn untuk membagi data menjadi data pelatihan dan pengujian.
from sklearn.neighbors import KNeighborsClassifier  # Mengimpor algoritma K-Nearest Neighbors (KNN) dari scikit-learn untuk model klasifikasi.
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score  # Mengimpor metrik evaluasi model
import matplotlib.pyplot as plt  # Mengimpor library matplotlib untuk visualisasi
import seaborn as sns  # Mengimpor library seaborn untuk visualisasi yang lebih menarik

# Load data  # Memuat dataset diabetes dari file CSV yang berisi informasi pasien dan hasil pemeriksaan diabetes.
data = pd.read_csv('diabetes.csv')  # Membaca data dari file 'diabetes.csv' menggunakan pandas.

X = data.drop('Outcome', axis=1)  # Membuat data prediktor (fitur) dengan menghapus kolom 'Outcome' dari dataset.
y = data['Outcome']  # Menentukan variabel target (klasifikasi) yang akan digunakan untuk model.

# Split data  # Memisahkan data menjadi dataset pelatihan (80%) dan pengujian (20%).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Menggunakan fungsi train_test_split dari scikit-learn untuk membagi data.

# Train model  # Memulai proses pelatihan model dengan menggunakan data pelatihan (X_train, y_train).
model = KNeighborsClassifier()  # Membuat objek model K-Nearest Neighbors (KNN).
model.fit(X_train, y_train)  # Melatih model menggunakan dataset pelatihan.

# Evaluate model  # Menilai performa model dengan menggunakan data pengujian (X_test, y_test).
y_pred = model.predict(X_test)  # Melakukan prediksi pada data pengujian
score = model.score(X_test, y_test)  # Menghitung akurasi model menggunakan dataset pengujian.

# Tampilkan evaluasi model
print("="*50)
print("MODEL EVALUATION")
print("="*50)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')  # Menampilkan akurasi model
print(f'Precision: {precision_score(y_test, y_pred):.4f}')  # Menampilkan precision
print(f'Recall: {recall_score(y_test, y_pred):.4f}')  # Menampilkan recall
print(f'F1-Score: {f1_score(y_test, y_pred):.4f}')  # Menampilkan F1-Score
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))  # Menampilkan classification report lengkap

# Confusion Matrix
print("="*50)
print("CONFUSION MATRIX")
print("="*50)
cm = confusion_matrix(y_test, y_pred)  # Membuat confusion matrix
print(cm)
print(f"\nTrue Negatives: {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")
print(f"True Positives: {cm[1][1]}")

# Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix - Diabetes Prediction Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')  # Menyimpan visualisasi
print("\nConfusion matrix visualization saved as 'confusion_matrix.png'")
plt.show()  # Menampilkan visualisasi

# Save model  # Menyimpan model yang telah dilatih untuk keperluan penggunaan kembali di kemudian hari.
with open('model.pkl', 'wb') as file:  # Membuka file 'model.pkl' dalam mode write-byte.
    pickle.dump(model, file)  # Menyimpan model menggunakan format Pickle.

# Save test data  # Menyimpan data hasil pengujian (X_test dan y_test) untuk referensi atau analisis lebih lanjut.
X_test.to_csv('X_test.csv', index=False)  # Menyimpan data X_test ke dalam file CSV.
y_test.to_csv('y_test.csv', index=False)  # Menyimpan data y_test ke dalam file CSV.
