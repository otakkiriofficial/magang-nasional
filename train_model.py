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
    skor_array = np.array(skor_array) 
    kondisi = [skor_array > 0, skor_array < 0]
    pilihan = ['positif', 'negatif']
    return np.select(kondisi, pilihan, default='netral')

def run_training_process():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Berkas dataset tidak ditemukan di: {DATA_PATH}. Pastikan Anda sudah menjalankan proses 'Pembersihan Relevancy'.")
        return

    with st.status("🚀 Melatih Model Sentimen Magang Nasional 2025...", expanded=True) as status:
        progress_bar = st.progress(0, text="Memulai proses pelatihan...")

        st.write("📂 Memuat Dataset Komentar...")
        progress_bar.progress(10)
        df_raw = pd.read_csv(DATA_PATH).dropna(subset=['textDisplay']).reset_index(drop=True)
        df = df_raw.copy()
        
        st.write("🧹 Membersihkan Teks (Cleansing + Stopwords + Stemming)...")
        progress_bar.progress(20)
        if 'clean_text' not in df.columns:
            df = run_preprocessing_pipeline(df, text_column='textDisplay')

        st.write("📖 Menghitung Skor Kamus Leksikon...")
        progress_bar.progress(40)
        lexicon = get_combined_lexicon()
        skor_leksikon = np.array([hitung_skor_leksikon(t, lexicon) for t in df['clean_text']]).reshape(-1, 1)

        st.write("🏷️ Melakukan Pelabelan Sentimen...")
        progress_bar.progress(50)
        y = pelabelan_otomatis(skor_leksikon.flatten())
        df['label_sentimen'] = y
        df['skor_leksikon'] = skor_leksikon.flatten() # Disimpan untuk visualisasi
        
        st.write(f"📊 Distribusi Dataset Awal: {pd.Series(y).value_counts().to_dict()}")

        # Filter kelas yang kurang dari 2
        valid_indices = df.groupby('label_sentimen').filter(lambda x: len(x) >= 2).index
        X_text_filtered = df.loc[valid_indices, 'clean_text']
        skor_leksikon_filtered = skor_leksikon[valid_indices]
        y_filtered = y[valid_indices]
        
        jumlah_kelas_valid = len(np.unique(y_filtered))
        st.write(f"📊 Distribusi Setelah Filter: {pd.Series(y_filtered).value_counts().to_dict()}")

        st.write("✂️ Pemisahan Data Training & Testing (Test Size: 30%)...")
        progress_bar.progress(60)
        X_tr_text, X_te_text, lex_tr, lex_te, y_tr, y_te = train_test_split(
            X_text_filtered, skor_leksikon_filtered, y_filtered, test_size=0.3, random_state=42, stratify=y_filtered
        )

        st.write("📐 Mengekstrak Fitur TF-IDF...")
        progress_bar.progress(70)
        vectorizer = TfidfVectorizer()
        X_tr_tfidf = vectorizer.fit_transform(X_tr_text)
        X_te_tfidf = vectorizer.transform(X_te_text)

        st.write("⚖️ Normalisasi Fitur Leksikon (MinMaxScaler)...")
        progress_bar.progress(75)
        scaler = MinMaxScaler()
        lex_tr_scaled = scaler.fit_transform(lex_tr)
        lex_te_scaled = scaler.transform(lex_te)

        st.write("🔗 Menggabungkan Matriks Fitur (TF-IDF + Leksikon)...")
        progress_bar.progress(80)
        X_tr_final = hstack([X_tr_tfidf, lex_tr_scaled])
        X_te_final = hstack([X_te_tfidf, lex_te_scaled])

        st.write("🤖 Melatih Algoritma Multinomial Naive Bayes...")
        progress_bar.progress(85)
        model = MultinomialNB(alpha=1)
        model.fit(X_tr_final, y_tr)

        st.write("🔄 Uji Evaluasi Klasifikasi...")
        progress_bar.progress(95)
        y_pred = model.predict(X_te_final)
        acc = metrics.accuracy_score(y_te, y_pred)
        
        st.success(f"🎯 **Akurasi Model Gabungan ({jumlah_kelas_valid} Kelas): {acc * 100:.2f}%**")
        
        st.write("📋 **Laporan Metrik:**")
        st.dataframe(pd.DataFrame(metrics.classification_report(y_te, y_pred, output_dict=True)).T.round(3))

        # =====================================================================
        # MODUL TRANSPARANSI 1-12 (MENGGUNAKAN DATA ASLI SISTEM)
        # =====================================================================
        st.markdown("---")
        st.markdown("### 🎓 Penelusuran Proses Step-by-Step (Validasi Data Asli)")
        
        with st.expander("1. 📂 Memuat Dataset Komentar", expanded=False):
            st.write("Sistem membaca file CSV asli dan membuang baris yang kosong (Missing Values). Berikut adalah 3 baris pertama dari data mentah:")
            st.dataframe(df_raw[['textDisplay']].head(3))

        with st.expander("2. 🧹 Membersihkan Teks (Cleansing + Stopwords + Stemming)", expanded=False):
            st.write("Membakukan teks agar bisa diolah komputer. Perhatikan perbedaan kolom teks asli dan teks yang sudah bersih di bawah ini:")
            st.dataframe(df[['textDisplay', 'clean_text']].head(3))

        with st.expander("3. 📖 Menghitung Skor Kamus Leksikon", expanded=False):
            st.write("Setiap kata dalam 'Teks Bersih' dicocokkan dengan kamus leksikon sentimen, lalu bobot angkanya dijumlahkan. Berikut hasil kalkulasi nyata dari sistem:")
            st.dataframe(df[['clean_text', 'skor_leksikon']].head(4))

        with st.expander("4. 🏷️ Melakukan Pelabelan Sentimen", expanded=False):
            st.write("Mengubah Skor Leksikon menjadi label kategori (Positif jika > 0, Negatif jika < 0, Netral jika = 0):")
            st.dataframe(df[['clean_text', 'skor_leksikon', 'label_sentimen']].head(4))

        with st.expander("5 & 6. 📊 Distribusi Dataset & Filter Kelas", expanded=False):
            st.write("**Distribusi Awal (Semua Kelas):**")
            st.json(pd.Series(y).value_counts().to_dict())
            st.write("**Distribusi Setelah Filter (Syarat Machine Learning: Minimal 2 data per kelas):**")
            st.info("Kelas dengan jumlah < 2 otomatis dibuang agar pembagian rasio data latih dan uji tidak error.")
            st.json(pd.Series(y_filtered).value_counts().to_dict())

        with st.expander("7. ✂️ Pemisahan Data Training & Testing (Test Size: 30%)", expanded=False):
            st.write("Dataset dipecah secara acak namun proporsional (stratified).")
            st.markdown(f"- **Data Latih (Training 70%)**: {X_tr_text.shape[0]} Baris (Digunakan untuk mengajari model)\n- **Data Uji (Testing 30%)**: {X_te_text.shape[0]} Baris (Disembunyikan untuk ujian)")
            st.write("Sampel Acak Data Uji:")
            st.dataframe(X_te_text.head(3))

        with st.expander("8. 📐 Mengekstrak Fitur TF-IDF", expanded=False):
            vocab = vectorizer.get_feature_names_out()
            st.write(f"Sistem mengekstrak **{len(vocab)} kosakata unik** dari seluruh Data Latih. Setiap kalimat kini diubah menjadi deretan angka (vektor matriks).")
            st.write("**Contoh 10 Kosakata Fitur yang terbentuk:**")
            st.code(", ".join(vocab[:10]))
            st.write(f"Dimensi Matriks TF-IDF Latih: **{X_tr_tfidf.shape[0]} Baris x {X_tr_tfidf.shape[1]} Kolom**")

        with st.expander("9. ⚖️ Normalisasi Fitur Leksikon (MinMaxScaler)", expanded=False):
            st.write("Skor leksikon memiliki rentang puluhan, sedangkan nilai TF-IDF hanya 0.0 - 1.0. Sistem melakukan normalisasi (0-1) pada skor leksikon agar skalanya seimbang.")
            df_norm = pd.DataFrame({
                'Skor Leksikon Latih Asli': lex_tr.flatten()[:5],
                'Skor Ter-Normalisasi (0 - 1)': lex_tr_scaled.flatten()[:5]
            })
            st.dataframe(df_norm)

        with st.expander("10. 🔗 Menggabungkan Matriks Fitur (TF-IDF + Leksikon)", expanded=False):
            st.write("Kolom Skor Leksikon Normalisasi ditempelkan di sebelah kanan Matriks TF-IDF menggunakan fungsi `hstack`.")
            st.markdown(f"- Matriks TF-IDF awal: {X_tr_tfidf.shape[1]} Fitur\n- Matriks + Leksikon Akhir: **{X_tr_final.shape[1]} Fitur (Bertambah 1 Kolom)**\n- **Dimensi Akhir Data Latih**: {X_tr_final.shape[0]} Baris x {X_tr_final.shape[1]} Kolom")

        with st.expander("11. 🤖 Melatih Algoritma Multinomial Naive Bayes", expanded=False):
            st.write("Model Naive Bayes mempelajari probabilitas kemunculan fitur pada setiap kelas sentimen.")
            classes = model.classes_
            class_counts = model.class_count_
            st.write("**Data Latih yang dipelajari Model:**")
            for c, count in zip(classes, class_counts):
                st.write(f"- Kelas **{c}**: Mempelajari {int(count)} dokumen.")
            st.write("Probabilitas fitur (*likelihood*) berhasil dihitung dan model kini siap membuat prediksi.")

        with st.expander("12. 🔄 Uji Evaluasi Klasifikasi", expanded=False):
            st.write("Model diminta menebak label dari Data Uji (yang tidak pernah ia lihat sebelumnya). Berikut perbandingan Prediksi Model dengan Kunci Jawaban Aslinya:")
            df_eval = pd.DataFrame({
                'Teks Data Uji (Disamarkan)': X_te_text.values[:5],
                'Kunci Jawaban Asli (Aktual)': y_te[:5],
                'Tebakan Model (Prediksi)': y_pred[:5]
            })
            # Mewarnai baris jika tebakannya benar/salah
            def highlight_correct(row):
                if row['Kunci Jawaban Asli (Aktual)'] == row['Tebakan Model (Prediksi)']:
                    return ['background-color: #d4edda'] * len(row)
                else:
                    return ['background-color: #f8d7da'] * len(row)
            
            st.dataframe(df_eval.style.apply(highlight_correct, axis=1))
        # =====================================================================

        st.write("💾 Menyimpan Komponen Model...")
        progress_bar.progress(100)
        with open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f: pickle.dump(vectorizer, f)
        with open(os.path.join(MODEL_DIR, 'lexicon_scaler.pkl'), 'wb') as f: pickle.dump(scaler, f)
        with open(os.path.join(MODEL_DIR, 'naive_bayes_model.pkl'), 'wb') as f: pickle.dump(model, f)
        with open(os.path.join(MODEL_DIR, 'lexicon_dict.pkl'), 'wb') as f: pickle.dump(lexicon, f)

        status.update(label=f"✅ Selesai! Model {jumlah_kelas_valid} Kelas Siap Digunakan.", state="complete", expanded=False)
    
    st.success("🎉 Proses ekstraksi dan pelatihan berhasil dituntaskan. Modul Prediksi dan Analisis sekarang siap digunakan.")

def show_training_page():
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
        3. **Klasifikasi Utama**: Mengklasifikasikan data menggunakan algoritma **Multinomial Naive Bayes**.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Mulai Kalibrasi & Pelatihan Model", type="primary", use_container_width=True):
        run_training_process()
