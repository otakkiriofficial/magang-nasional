import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from scipy.sparse import hstack

from preprocessing import run_preprocessing_pipeline
from utils import get_combined_lexicon, hitung_skor_leksikon

# Konfigurasi path dataset hasil filtering
DATA_PATH = os.path.join('data', 'data_paper.csv')
MODEL_DIR = 'models'

def pelabelan_otomatis(skor_array):
    """Melabeli dataset murni berdasarkan skor mutlak menggunakan np.select."""
    skor_array = np.array(skor_array) # Pastikan formatnya numpy array
    kondisi = [skor_array > 0, skor_array < 0]
    pilihan = ['positif', 'negatif']
    
    # Jika tidak memenuhi kedua kondisi di atas, jadikan 'netral'
    return np.select(kondisi, pilihan, default='netral')

def run_training_process():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Berkas dataset tidak ditemukan di: {DATA_PATH}. Pastikan Anda sudah menjalankan proses 'Pembersihan Relevancy'.")
        return

    with st.status("🚀 Melatih Model Sentimen Magang Nasional 2025...", expanded=True) as status:
        progress_bar = st.progress(0, text="Memulai proses pelatihan...")

        st.write("📂 Memuat Dataset Komentar...")
        progress_bar.progress(10, text="Tahap 1/8: Memuat Dataset...")
        df = pd.read_csv(DATA_PATH).dropna(subset=['textDisplay']).reset_index(drop=True)
        
        st.write("🧹 Membersihkan Teks (Cleansing + Stopwords + Stemming)...")
        progress_bar.progress(20, text="Tahap 2/8: Preprocessing Teks (Ini memakan waktu, mohon tunggu)...")
        if 'clean_text' not in df.columns:
            df = run_preprocessing_pipeline(df, text_column='textDisplay')

        st.write("📖 Menghitung Skor Kamus Leksikon...")
        progress_bar.progress(40, text="Tahap 3/8: Ekstraksi Skor Leksikon...")
        lexicon = get_combined_lexicon()
        skor_leksikon = np.array([hitung_skor_leksikon(t, lexicon) for t in df['clean_text']]).reshape(-1, 1)

        st.write("🏷️ Melakukan Pelabelan Sentimen...")
        progress_bar.progress(50, text="Tahap 4/8: Pelabelan Otomatis (Termasuk Netral)...")
        # Semua data digunakan (Multikelas: Positif, Negatif, Netral)
        y = pelabelan_otomatis(skor_leksikon.flatten())
        df['label_sentimen'] = y
        
        X_text = df['clean_text']
        st.write(f"📊 Distribusi Dataset (3 Kelas): {pd.Series(y).value_counts().to_dict()}")

        # Filter out classes with fewer than 2 members
        valid_indices = df.groupby('label_sentimen').filter(lambda x: len(x) >= 2).index
        
        # Apply the filter to all variables
        X_text_filtered = df.loc[valid_indices, 'clean_text']
        skor_leksikon_filtered = skor_leksikon[valid_indices]
        y_filtered = y[valid_indices]

        st.write("✂️ Pemisahan Data Training & Testing (Test Size: 30%)...")
        X_tr_text, X_te_text, lex_tr, lex_te, y_tr, y_te = train_test_split(
            X_text_filtered, skor_leksikon_filtered, y_filtered, test_size=0.3, random_state=42, stratify=y_filtered
        )

        st.write("📐 Mengekstrak Fitur TF-IDF...")
        progress_bar.progress(70, text="Tahap 5/8: Vektorisasi TF-IDF & Normalisasi Leksikon...")
        # Menggunakan TfidfVectorizer utuh tanpa batas max_features
        vectorizer = TfidfVectorizer()
        X_tr_tfidf = vectorizer.fit_transform(X_tr_text)
        X_te_tfidf = vectorizer.transform(X_te_text)

        st.write("⚖️ Normalisasi Fitur Leksikon (MinMaxScaler)...")
        scaler = MinMaxScaler()
        lex_tr_scaled = scaler.fit_transform(lex_tr)
        lex_te_scaled = scaler.transform(lex_te)

        st.write("🔗 Menggabungkan Matriks Fitur (TF-IDF + Leksikon)...")
        X_tr_final = hstack([X_tr_tfidf, lex_tr_scaled])
        X_te_final = hstack([X_te_tfidf, lex_te_scaled])

        st.write("🤖 Melatih Algoritma Multinomial Naive Bayes...")
        progress_bar.progress(85, text="Tahap 6/8: Melatih Model Naive Bayes...")
        model = MultinomialNB(alpha=1)
        model.fit(X_tr_final, y_tr)

        st.write("🔄 Uji Evaluasi Klasifikasi...")
        progress_bar.progress(95, text="Tahap 7/8: Evaluasi Akurasi...")
        y_pred = model.predict(X_te_final)
        acc = metrics.accuracy_score(y_te, y_pred)
        st.write(f"🎯 **Akurasi Model Gabungan (3 Kelas): {acc * 100:.2f}%**")
        
        st.write("📋 **Laporan Metrik:**")
        st.dataframe(pd.DataFrame(metrics.classification_report(y_te, y_pred, output_dict=True)).T.round(3))

        st.write("💾 Menyimpan Komponen Model...")
        progress_bar.progress(100, text="Tahap 8/8: Menyimpan Model... Selesai!")
        with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f: pickle.dump(vectorizer, f)
        with open(os.path.join(MODEL_DIR, 'lexicon_scaler.pkl'), 'wb') as f: pickle.dump(scaler, f)
        with open(os.path.join(MODEL_DIR, 'naive_bayes_model.pkl'), 'wb') as f: pickle.dump(model, f)
        with open(os.path.join(MODEL_DIR, 'lexicon_dict.pkl'), 'wb') as f: pickle.dump(lexicon, f)

        status.update(label="✅ Selesai! Model 3 Kelas Siap Digunakan.", state="complete", expanded=False)
    
    st.success("🎉 Proses ekstraksi dan pelatihan multikelas berhasil dituntaskan. Modul Prediksi dan Analisis sekarang siap digunakan.")

def show_training_page():
    # Header yang disesuaikan dengan UI yang profesional
    st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #1B4965; margin-bottom: 25px">
            <h2 style="color: #1B4965; margin: 0;">⚙️ Pelatihan Model Sentimen</h2>
            <p style="color: #62B6CB; margin: 0;">Kalibrasi Algoritma Naïve Bayes untuk Program Magang Nasional 2025</p>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("📌 Arsitektur Komputasi Machine Learning", expanded=True):
        st.markdown("""
        **Alur yang akan dieksekusi oleh sistem:**
        1. **Pra-pemrosesan Data**: Penerapan teknik *Cleansing*, *Case Folding*, penghapusan *Stopwords*, dan *Stemming* menggunakan pustaka Sastrawi.
        2. **Pendekatan Hibrida (Vektor & Semantik)**: Model menggabungkan **Vektor Statistik (TF-IDF Murni)** dengan **Pengetahuan Semantik (Skor Leksikon)** yang telah dinormalisasi.
        3. **Klasifikasi Utama**: Mengklasifikasikan data ke dalam **3 Kelas (Positif, Negatif, Netral)** menggunakan algoritma **Multinomial Naive Bayes**.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tombol eksekusi yang dibuat lebih lebar dan menonjol
    if st.button("🚀 Mulai Kalibrasi & Pelatihan Model", type="primary", use_container_width=True):
        run_training_process()