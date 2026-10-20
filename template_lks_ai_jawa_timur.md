# LOMBA KOMPETENSI SISWA (LKS) SMK TINGKAT PROVINSI JAWA TIMUR
## BIDANG LOMBA: ARTIFICIAL INTELLIGENCE (AI)
**Nomor Peserta / Nama Siswa:** [Isi Nama Peserta]  
**Asal Sekolah:** [Isi Nama Sekolah]  

---

## Modul 1: Import Library & Loading Dataset
*Tujuan: Memuat seluruh library esensial dan membaca dataset yang diberikan ke dalam memori.*

```python
# Import library dasar untuk manipulasi data dan visualisasi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi visualisasi
sns.set(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

# Load dataset (Sesuaikan path file soal LKS)
df = pd.read_dataset_atau_csv('nama_dataset.csv')

# Menampilkan 5 baris pertama
print("--- 5 Baris Pertama Dataset ---")
display(df.head())

# Memeriksa informasi umum dataset (tipe data & jumlah baris/kolom)
print("\n--- Informasi Dataset ---")
df.info()
```

---

## Modul 2: Exploratory Data Analysis (EDA)
*Tujuan: Menganalisis karakteristik data, mendeteksi missing values, outliers, serta korelasi antar fitur.*

```python
# 1. Cek Missing Values
print("--- Jumlah Missing Values per Kolom ---")
missing_val = df.isnull().sum()
print(missing_val[missing_val > 0])

# 2. Statistik Deskriptif untuk Data Numerik
print("\n--- Statistik Deskriptif ---")
display(df.describe())

# 3. Visualisasi Distribusi Target Variabel (Contoh: Kolom 'target')
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df) # Ganti dengan tipe plot yang sesuai jika regresi (sns.histplot)
plt.title("Distribusi Target Variabel")
plt.show()

# 4. Heatmap Korelasi (Hanya untuk fitur numerik)
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriks Korelasi Fitur Numerik")
plt.show()
```

---

## Modul 3: Data Preprocessing & Feature Engineering
*Tujuan: Membersihkan data, menangani missing values, melakukan encoding, scaling, dan pembagian dataset.*

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Menangani Missing Values (Contoh: Imputasi Mean untuk Numerik)
# numeric_cols = df.select_dtypes(include=[np.number]).columns
# imputer = SimpleImputer(strategy='mean')
# df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# 2. Enkoding Fitur Kategorikal (Jika ada)
# label_enc = LabelEncoder()
# df['kolom_kategori'] = label_enc.fit_transform(df['kolom_kategori'])

# 3. Memisahkan Fitur (X) dan Target (y)
X = df.drop(columns=['target']) # Sesuaikan nama kolom target
y = df['target']

# 4. Train-Test Split (Rasio standar 80:20 atau sesuai ketentuan soal)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) < 20 else None
)

print(f"Dimensi X_train: {X_train.shape}")
print(f"Dimensi X_test: {X_test.shape}")

# 5. Feature Scaling (Opsional tapi direkomendasikan untuk model tertentu)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Modul 4: Pemodelan Machine Learning
*Tujuan: Melatih model algoritma baseline dan melakukan optimasi hyperparameter.*

```python
from sklearn.ensemble import RandomForestClassifier # Contoh model klasifikasi
from sklearn.metrics import accuracy_score, classification_report

# Inisialisasi dan Training Model Baseline
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Prediksi pada Data Train dan Test (Untuk deteksi Overfitting)
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("Akurasi pada Data Train:", accuracy_score(y_train, y_train_pred))
print("Akurasi pada Data Test: ", accuracy_score(y_test, y_test_pred))
```

---

## Modul 5: Evaluasi Model & Kesimpulan
*Tujuan: Menilai performa model secara komprehensif menggunakan metrik evaluasi yang sesuai.*

```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Laporan Klasifikasi Lengkap (Precision, Recall, F1-Score)
print("--- Laporan Klasifikasi (Test Set) ---")
print(classification_report(y_test, y_test_pred))

# 2. Visualisasi Confusion Matrix
plt.figure(figsize=(6, 6))
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
```