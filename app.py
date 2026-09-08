import streamlit as st
import os
import preprocessing
from utils import load_models
from views import home, scraping, prediction, analysis, preprocessing
import train_model

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="InternSphere 2025",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS KUSTOM UNTUK NAVBAR MODERN ---
st.markdown("""
    <style>
    /* Sembunyikan Header Bawaan Streamlit */
    header {visibility: hidden;}
    .main .block-container {padding-top: 0rem;}

    /* Container Utama Navbar */
    .nav-wrapper {
        background-color: #1B4965;
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 4px solid #62B6CB;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .nav-brand {
        color: white;
        font-weight: bold;
        font-size: 1.5rem;
    }

    /* Styling Radio Button Agar Menjadi Navbar */
    div[data-testid="stHorizontalBlock"] {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Menghilangkan bulatan radio dan merapikan label */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: 500;
    }

    /* Membuat tombol navigasi terlihat proporsional */
    .st-emotion-cache-16idsys p {
        color: #1B4965;
    }
    
    /* Hover effect pada menu */
    button[kind="secondary"] {
        border-radius: 20px;
        border: 1px solid #1B4965;
        color: #1B4965;
        transition: 0.3s;
    }
    
    button[kind="secondary"]:hover {
        background-color: #1B4965;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. HEADER IDENTITAS (STATIS) ---
st.markdown("""
    <div class="nav-wrapper">
        <div class="nav-brand">Analisis Sentimen Masyarakat Terhadap Program Magang Nasional 2025 Menggunakan Metode Naive Bayes</div>
    </div>
""", unsafe_allow_html=True)

# --- 4. NAVIGASI HORIZONTAL ---
# Menggunakan 6 kolom agar tombol berjajar rapi sesuai urutan alur kerja
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    btn_home = st.button("🏠 Beranda", use_container_width=True)
with col2:
    btn_scrap = st.button("📥 Ambil Data", use_container_width=True)
with col3:
    btn_clean = st.button("🧹 Pembersihan", use_container_width=True)
with col4:
    btn_train = st.button("⚙️ Training Model", use_container_width=True)
with col5:
    btn_pred = st.button("⚡ Prediksi Cepat", use_container_width=True)
with col6:
    btn_anal = st.button("📊 Laporan Analisis", use_container_width=True)

# --- 5. LOGIKA ROUTING (SESSION STATE) ---
if 'page' not in st.session_state:
    st.session_state.page = "Beranda"

if btn_home: st.session_state.page = "Beranda"
if btn_scrap: st.session_state.page = "Ambil Data/Scraping"
if btn_clean: st.session_state.page = "Pembersihan Relevancy"
if btn_train: st.session_state.page = "Training Model"
if btn_pred: st.session_state.page = "Prediksi Kata Cepat"
if btn_anal: st.session_state.page = "Laporan Analisis Model"

# --- 6. LOAD MODELS ---
model, vectorizer = load_models()

# --- 7. ROUTING TAMPILAN HALAMAN ---
st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.page == "Beranda":
    # Ganti dengan pemanggilan fungsi untuk halaman Beranda Anda
    home.show_home_page()

elif st.session_state.page == "Ambil Data/Scraping":
    # Ganti dengan pemanggilan fungsi untuk halaman Scraping
    scraping.show_scraping_page()

elif st.session_state.page == "Pembersihan Relevancy":
    # Pastikan Anda membuat file/fungsi baru untuk halaman ini
    preprocessing.show_preprocessing_page()

elif st.session_state.page == "Training Model":
    train_model.show_training_page()

elif st.session_state.page == "Prediksi Kata Cepat":
    if model is not None and vectorizer is not None:
        prediction.show_prediction_page(model, vectorizer)
    else:
        st.warning("⚠️ Model belum siap. Silakan selesaikan proses Training Model terlebih dahulu.")

elif st.session_state.page == "Laporan Analisis Model":
    if model is not None and vectorizer is not None:
        analysis.show_analysis_page(model, vectorizer)
    else:
        st.error("❌ Akses ditolak. Model AI tidak ditemukan untuk dianalisis.")