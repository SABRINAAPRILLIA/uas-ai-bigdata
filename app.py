import streamlit as st
import joblib
import numpy as np

# 1. Load model, scaler, dan encoders yang sudah dilatih di Colab
model = joblib.load("car_price_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Prediksi Harga Mobil Bekas", layout="centered")

st.title("🚗 Aplikasi Prediksi Harga Mobil Bekas")
st.write("Project Akhir AI & Big Data 2026")

# ---- SIDEBAR: MENAMPILKAN METRIK PERFORMA MODEL (WAJIB UAS) ----
st.sidebar.header("📊 Performa Model (Random Forest)")
st.sidebar.markdown("""
Model terbaik yang dipilih untuk deployment adalah **Random Forest Regressor** dengan hasil evaluasi data testing:
- **R² Score:** 0.7811 (78.1%)
- **MAE:** $6,563.88
- **RMSE:** $9,794.38
""")

st.write("---")

# ---- MAIN INTERFACE: INPUT PARAMETER MINIMAL 4 (WAJIB UAS) ----
st.subheader("Masukkan Spesifikasi Mobil:")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Merek Mobil (Brand)", encoders['brand'].classes_)
    model_car = st.selectbox("Model Mobil", encoders['model'].classes_)
    tahun = st.number_input("Tahun Pembuatan (Model Year)", 2000, 2026, 2020)
    milage = st.number_input("Jarak Tempuh (Milage dalam Mil)", 0, 500000, 45000)

with col2:
    fuel_type = st.selectbox("Tipe Bahan Bakar (Fuel Type)", encoders['fuel_type'].classes_)
    transmission = st.selectbox("Jenis Transmisi", encoders['transmission'].classes_)
    accident = st.selectbox("Riwayat Kecelakaan (Accident)", encoders['accident'].classes_)
    clean_title = st.selectbox("Status Dokumen (Clean Title)", encoders['clean_title'].classes_)

# Nilai default internal untuk fitur sisa yang tidak masuk UI utama agar dimensi array tetap 12 kolom
engine_default = encoders['engine'].transform([encoders['engine'].classes_[0]])[0]
ext_col_default = encoders['ext_col'].transform([encoders['ext_col'].classes_[0]])[0]
int_col_default = encoders['int_col'].transform([encoders['int_col'].classes_[0]])[0]

# ---- PROSES PREDIKSI ----
if st.button("Hitung Estimasi Harga", type="primary"):
    # Hitung umur mobil secara otomatis (Feature Engineering)
    car_age = 2025 - tahun

    # Transformasi input teks dari UI menjadi angka menggunakan LabelEncoder yang di-load
    brand_enc = encoders['brand'].transform([brand])[0]
    model_enc = encoders['model'].transform([model_car])[0]
    fuel_enc = encoders['fuel_type'].transform([fuel_type])[0]
    trans_enc = encoders['transmission'].transform([transmission])[0]
    accident_enc = encoders['accident'].transform([accident])[0]
    title_enc = encoders['clean_title'].transform([clean_title])[0]

    # Susun struktur data 12 fitur presisi sesuai urutan kolom waktu training
    input_data = np.array([[
        brand_enc, model_enc, tahun, milage, fuel_enc,
        engine_default, trans_enc, ext_col_default, int_col_default,
        accident_enc, title_enc, car_age
    ]])

    # Standardisasi data lewat scaler yang di-load
    input_scaled = scaler.transform(input_data)

    # Menghitung prediksi harga mobil
    harga_prediksi = model.predict(input_scaled)

    # Tampilkan hasil akhir ke layar web
    st.success(f"### 💵 Estimasi Harga Jual: ${harga_prediksi[0]:,.2f}")
