import streamlit as st
import os

def show_home_page():
    # --- CUSTOM CSS UNTUK TAMPILAN PROFESIONAL ---
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.8rem;
            color: #1B4965;
            font-weight: 800;
            text-align: center;
            margin-bottom: 5px;
            letter-spacing: 1px;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #62B6CB;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 500;
            line-height: 1.5;
        }
        .feature-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            border-top: 5px solid #1B4965;
            height: 100%;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .feature-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.1);
        }
        .feature-title {
            color: #1B4965;
            font-size: 1.1rem;
            font-weight: bold;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .feature-desc {
            font-size: 0.95rem;
            color: #555;
            line-height: 1.4;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- HEADER ---
    st.markdown('<p class="main-header">Adinda Permata Sari</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analisis Sentimen Masyarakat Terhadap Program Magang Nasional 2025 Menggunakan Metode Naive Bayes<br>NPM 22441207</p>', unsafe_allow_html=True)

    # --- CEK KESIAPAN MODEL ---
    # Memeriksa eksistensi model baik di root direktori maupun di dalam folder 'models'
    model_paths = ['model_naive_bayes.pkl', 'models/model_naive_bayes.pkl', 'models/naive_bayes_model.pkl']
    vector_paths = ['tfidf_vectorizer.pkl', 'models/tfidf_vectorizer.pkl']
    
    model_exists = any(os.path.exists(p) for p in model_paths)
    vector_exists = any(os.path.exists(p) for p in vector_paths)

    if not (model_exists and vector_exists):
        st.error("⚠️ **Sistem Belum Terkalibrasi:** File Model AI belum dibuat. Silakan jalankan proses pada menu **Training Model** terlebih dahulu.")
    else:
        st.success("✅ **Sistem Siap Beroperasi:** Engine Naïve Bayes dan TF-IDF Vectorizer telah berhasil dimuat.")

    st.markdown("---")
    st.markdown("<h4 style='text-align: center; color: #1B4965; margin-bottom: 25px;'>Alur Kerja Sistem (Pipeline)</h4>", unsafe_allow_html=True)

    # --- LAYOUT FITUR KARTU (BARIS 1) ---
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">📥 1. Ambil Data</div>
            <p class="feature-desc">Modul scraping untuk mengumpulkan opini publik secara langsung dari komentar YouTube API terkait topik Magang Nasional 2025.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">🧹 2. Pembersihan Relevancy</div>
            <p class="feature-desc">Fase penyaringan (filtering) dataset mentah untuk membuang komentar spam, iklan, dan opini yang tidak memiliki relevansi konteks.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">⚙️ 3. Training Model</div>
            <p class="feature-desc">Mengeksekusi pipeline NLP, melakukan pembobotan term menggunakan TF-IDF, dan melatih probabilitas algoritma Naïve Bayes.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- LAYOUT FITUR KARTU (BARIS 2 - DITENGAHKAN) ---
    c_spacer1, c4, c5, c_spacer2 = st.columns([1, 2, 2, 1])
    
    with c4:
        st.markdown("""
        <div class="feature-box" style="border-top-color: #62B6CB;">
            <div class="feature-title">⚡ 4. Prediksi Cepat</div>
            <p class="feature-desc">Pengujian sentimen *real-time* pada teks tunggal yang menampilkan rincian komputasi matematis (*white-box testing*) dari Naïve Bayes.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c5:
        st.markdown("""
        <div class="feature-box" style="border-top-color: #62B6CB;">
            <div class="feature-title">📊 5. Laporan Analisis</div>
            <p class="feature-desc">Dashboard komprehensif untuk memantau metrik evaluasi model (Confusion Matrix) dan visualisasi WordCloud asosiasi sentimen.</p>
        </div>
        """, unsafe_allow_html=True)