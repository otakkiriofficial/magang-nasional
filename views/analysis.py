from collections import Counter
import os
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from .nlp_engine import run_preprocessing_pipeline
from scipy.sparse import csr_matrix, hstack
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
from wordcloud import WordCloud

# Mengambil KATA_NEGASI dari modul acuan
from utils import KATA_NEGASI

# --- 1. KONFIGURASI VISUAL & HELPER ---
BLACKLIST = [""]
MODEL_DIR = "models"


def _load_analysis_components():
    """Memuat 4 komponen file AI secara mandiri"""
    cache = {}
    files = [
        ("tfidf_vectorizer.pkl", "vec"),
        ("lexicon_scaler.pkl", "scaler"),
        ("naive_bayes_model.pkl", "model"),
        ("lexicon_dict.pkl", "lex"),
    ]
    for fname, key in files:
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            cache[key] = pickle.load(f)
    return cache


def extract_lexicon_visuals(text, lexicon_dict):
    """Logika ekstraksi visualisasi kata dan penghitungan skor mentah"""
    if pd.isna(text) or str(text).strip() == "":
        return "", "", 0.0, "Netral"

    words = str(text).split()
    pos_found = []
    neg_found = []
    total_score = 0.0
    negasi_aktif = False

    for word in words:
        if word in KATA_NEGASI:
            negasi_aktif = True
            continue

        if word in lexicon_dict:
            bobot = lexicon_dict[word]
            display_word = word

            if negasi_aktif:
                bobot *= -1
                display_word = f"NOT_{word}"
                negasi_aktif = False  

            total_score += bobot

            if bobot > 0:
                pos_found.append(f"{display_word}(+{int(bobot)})")
            elif bobot < 0:
                neg_found.append(f"{display_word}({int(bobot)})")

    if total_score > 0:
        label = "Positif"
    elif total_score < 0:
        label = "Negatif"
    else:
        label = "Netral"

    return (
        " ".join(pos_found),
        " ".join(neg_found),
        round(total_score, 2),
        label,
    )


# --- 2. FUNGSI VIEW UTAMA ---
def show_analysis_page(model=None, vectorizer=None):
    st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #1B4965; margin-bottom: 25px">
            <h2 style="color: #1B4965; margin: 0;">📊 Laporan Analisis Sentimen</h2>
            <p style="color: #62B6CB; margin: 0;">Dashboard Interaktif Program Magang Nasional 2025: Distribusi, Visualisasi, dan Laporan Klasifikasi</p>
        </div>
    """, unsafe_allow_html=True)

    comps = _load_analysis_components()

    if comps is None:
        st.warning(
            "⚠️ **File Model Belum Lengkap:** Sistem tidak menemukan 4 file model utama. Silakan lakukan 'Training Model' terlebih dahulu."
        )
        return

    st.markdown("### 📂 Input Dataset Uji")
    uploaded_file = st.file_uploader(
        "Unggah Dataset Komentar Magang Nasional (Format: CSV)", type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "textDisplay" not in df.columns:
            st.error(
                "❌ **Format Tidak Sesuai:** Kolom 'textDisplay' tidak ditemukan pada file CSV."
            )
            return

        progress_bar = st.progress(0, text="Memulai proses analisis data...")

        progress_bar.progress(20, text="⏳ 1/4: Membersihkan dan menormalisasi teks (Preprocessing)...")
        df = run_preprocessing_pipeline(df, text_column="textDisplay")
        lexicon_dict = comps["lex"]

        progress_bar.progress(50, text="⏳ 2/4: Mengekstraksi dan mencocokkan kata dengan Kamus Leksikon...")
        df[
            [
                "kata_positif",
                "kata_negatif",
                "skor_sentimen",
                "lexicon_label",
            ]
        ] = df["clean_text"].apply(
            lambda x: pd.Series(extract_lexicon_visuals(x, lexicon_dict))
        )

        progress_bar.progress(75, text="⏳ 3/4: Melakukan vektorisasi TF-IDF dan memprediksi sentimen (Naïve Bayes)...")
        skor_mentah_arr = np.array(df["skor_sentimen"]).reshape(-1, 1)
        skor_scaled = comps["scaler"].transform(skor_mentah_arr)

        tfidf_vectors = comps["vec"].transform(df["clean_text"].astype(str))
        final_features = hstack([tfidf_vectors, csr_matrix(skor_scaled)])

        df["prediction"] = comps["model"].predict(final_features)
        df["prediction"] = df["prediction"].astype(str).str.strip().str.capitalize() 
        df["confidence"] = comps["model"].predict_proba(final_features).max(axis=1)

        progress_bar.progress(100, text="✅ 4/4: Proses selesai! Mempersiapkan visualisasi...")

        st.success(
            f"✅ Analisis berhasil diselesaikan untuk **{len(df)}** baris ulasan bersih terkait Program Magang Nasional."
        )

        # Definisi warna seragam untuk semua grafik
        color_map = {
            "Positif": "#2563EB", # Biru Terang
            "Negatif": "#DC2626", # Merah Terang
            "Netral": "#D97706",  # Oranye
        }

        tabs = st.tabs(
            [
                "📈 Distribusi Sentimen",
                "☁️ WordCloud",
                "📑 Laporan per Sentimen",
                "📝 Tabulasi Hasil",
                "🎯 Evaluasi Model",
            ]
        )

        # TAB 1: PIE CHART
        with tabs[0]:
            st.markdown("#### 📊 Proporsi Sentimen Masyarakat")
            st.caption(
                "Visualisasi distribusi sentimen hasil klasifikasi menggunakan algoritma Naïve Bayes."
            )

            pred_counts = df["prediction"].value_counts().reset_index()
            pred_counts.columns = ["prediction", "count"]

            total_data = pred_counts["count"].sum()

            fig_pie = px.pie(
                pred_counts,
                values="count",
                names="prediction",
                hole=0.55,
                color="prediction",
                color_discrete_map={
                    "Positif": "#36CFC9",
                    "Netral": "#FAAD14",
                    "Negatif": "#FF4D4F"
                }
            )

            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent+label",
                pull=[0.03, 0.03, 0.03],  # efek sedikit keluar
                marker=dict(
                    line=dict(color="white", width=3)
                ),
                hovertemplate=
                "<b>%{label}</b><br>" +
                "Jumlah: %{value}<br>" +
                "Persentase: %{percent}<extra></extra>"
            )

            fig_pie.update_layout(
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.15,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(t=50, b=50, l=20, r=20),
                annotations=[
                    dict(
                        text=f"<b>{total_data}</b><br>Total Data",
                        x=0.5,
                        y=0.5,
                        font=dict(size=20),
                        showarrow=False
                    )
                ]
            )

            st.plotly_chart(fig_pie, use_container_width=True)
         
        # TAB 2: WORDCLOUD (DIPERBAIKI)
        with tabs[1]:
            st.markdown("#### ☁️ Visualisasi Asosiasi Kata")
            cols = st.columns(3)
            conf_wc = [("Negatif", cols[0], "autumn", "🔴 Indikasi Negatif", "kata_negatif"),
                       ("Netral", cols[1], "Set2", "🟡 Indikasi Netral", "clean_text"),
                       ("Positif", cols[2], "cool", "🔵 Indikasi Positif", "kata_positif")]
            for sent, col, cmap_name, label, text_source in conf_wc:
                with col:
                    st.markdown(f"<h5 style='text-align:center'>{label}</h5>", unsafe_allow_html=True)
                    data_sent = df[df["prediction"] == sent]
                    text_all = " ".join(data_sent[text_source].astype(str))
                    text_all = re.sub(r"\([+-]?\d+\)", "", text_all).replace("NOT_", "")
                    if text_all.strip():
                        wc = WordCloud(width=800, height=800, background_color="white", colormap=cmap_name, max_words=100).generate(text_all)
                        fig, ax = plt.subplots()
                        ax.imshow(wc, interpolation="bilinear")
                        ax.axis("off")
                        st.pyplot(fig)
                        plt.close(fig)
                             
        # TAB 3: BAR CHART & TABEL 
        with tabs[2]:
            st.markdown("#### 📑 Laporan Analisis Kuantitatif per Sentimen")
            st.caption("Memvisualisasikan volume opini dalam bentuk diagram batang beserta rincian komentar spesifik.")
            
            # --- TAMPILAN BAR CHART ---
            bar_counts = df["prediction"].value_counts().reset_index()
            bar_counts.columns = ["Sentimen", "Jumlah Komentar"]
            
            fig_bar = px.bar(
                bar_counts,
                x="Sentimen",
                y="Jumlah Komentar",
                color="Sentimen",
                color_discrete_map=color_map,
                text="Jumlah Komentar",
                title="Diagram Batang Frekuensi Sentimen"
            )
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")
            
            # --- TAMPILAN TABEL FILTER ---
            st.markdown("##### 🔍 Rincian Data Ulasan")
            target = st.radio(
                "Filter Data Berdasarkan Kelas Sentimen:",
                ["Positif", "Negatif", "Netral"],
                horizontal=True,
            )
            
            df_target = df[df["prediction"] == target].copy()
            st.write(f"Menampilkan **{len(df_target)}** komentar dengan sentimen **{target}**.")
            
            if not df_target.empty:
                df_show = df_target[["textDisplay", "clean_text", "confidence"]].copy()
                df_show.rename(columns={
                    "textDisplay": "Teks Asli (Komentar YouTube)",
                    "clean_text": "Teks Hasil Preprocessing",
                    "confidence": "Keyakinan Naïve Bayes"
                }, inplace=True)
                
                df_show["Keyakinan Naïve Bayes"] = pd.to_numeric(
                    df_show["Keyakinan Naïve Bayes"], errors='coerce'
                ).apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
                
                st.dataframe(df_show, use_container_width=True, hide_index=True)
            else:
                st.info(f"Tidak ada data komentar yang terklasifikasi sebagai sentimen {target}.")

        with tabs[3]:
            st.markdown("#### 📝 Tabulasi Detail Hasil Prediksi Model vs Leksikon")
            st.caption("Menampilkan perbandingan komprehensif antara ekstraksi kata, perhitungan bobot Leksikon, dan hasil prediksi akhir algoritma Naïve Bayes.")
            
            cols_show = [
                "textDisplay", "case_folding", "cleaning", "tokenizing", 
                "normalizing", "filtering", "stemming", "clean_text",
                "kata_positif", "kata_negatif", "skor_sentimen",
                "lexicon_label", "prediction", "confidence"
            ]
            
            existing_cols = [col for col in cols_show if col in df.columns]
            df_display = df[existing_cols].copy()
            
            if "kata_positif" in df_display.columns:
                df_display["kata_positif"] = df_display["kata_positif"].astype(str).str.replace("NOT_", "", regex=False)
            if "kata_negatif" in df_display.columns:
                df_display["kata_negatif"] = df_display["kata_negatif"].astype(str).str.replace("NOT_", "", regex=False)

            df_display.rename(columns={
                "textDisplay": "Teks Asli (textDisplay)",
                "clean_text": "Teks Preprocessing Lengkap",
                "case_folding": "Case Folding",
                "cleaning": "Cleaning",
                "tokenizing": "Tokenizing",
                "normalizing": "Normalizing",
                "filtering": "Filtering",
                "stemming": "Stemming",
                "kata_positif": "Kata Positif",
                "kata_negatif": "Kata Negatif",
                "skor_sentimen": "Skor Labeling",
                "lexicon_label": "Label Leksikon",
                "prediction": "Prediksi Naïve Bayes",
                "confidence": "Confidence Score"
            }, inplace=True)

            if "Confidence Score" in df_display.columns:
                df_display["Confidence Score"] = pd.to_numeric(
                    df_display["Confidence Score"], errors='coerce'
                ).apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")

            st.dataframe(
                df_display, 
                use_container_width=True,
                height=500,        
                hide_index=True    
            )

            st.markdown("---")
            st.markdown("##### 📥 Ekspor Hasil Analisis")
            
            csv_data = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Tabulasi Lengkap (Format CSV)",
                data=csv_data,
                file_name="hasil_analisis_magang_nasional_2025.csv",
                mime="text/csv",
                use_container_width=True
            )

        with tabs[4]:
            st.markdown("#### 🎯 Matriks Evaluasi Kinerja Algoritma")
            
            KOLOM_AKTUAL = "lexicon_label" 
            
            valid_labels = ["Negatif", "Netral", "Positif"]
            df[KOLOM_AKTUAL] = df[KOLOM_AKTUAL].astype(str).str.strip().str.capitalize()
            eval_df = df[df[KOLOM_AKTUAL].isin(valid_labels)].dropna(subset=[KOLOM_AKTUAL])

            if not eval_df.empty:
                y_true = eval_df[KOLOM_AKTUAL] 
                y_pred = eval_df["prediction"] 
                
                labels = sorted(list(set(y_true) | set(y_pred)))
                acc = accuracy_score(y_true, y_pred)
                report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

                st.markdown(f"**Akurasi Keseluruhan Model:** <span style='font-size: 20px; color: #1B4965; font-weight: bold;'>{acc:.2%}</span>", unsafe_allow_html=True)
                st.markdown("<br>**Tabel Rincian Metrik Klasifikasi:**", unsafe_allow_html=True)
                
                report_df = pd.DataFrame(report).transpose()
                report_df.rename(columns={
                    'precision': 'Presisi (Precision)', 
                    'recall': 'Sensitivitas (Recall)', 
                    'f1-score': 'F1-Score', 
                    'support': 'Jumlah Data (Support)'
                }, inplace=True)
                
                st.table(report_df.style.format({
                    'Presisi (Precision)': '{:.3f}', 'Sensitivitas (Recall)': '{:.3f}',
                    'F1-Score': '{:.3f}', 'Jumlah Data (Support)': '{:.0f}'
                }))
                
                st.divider()
                st.markdown("**Confusion Matrix (Matriks Kebingungan):**")
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                fig_cm, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, linewidths=1, linecolor="black")
                plt.ylabel("Label Aktual (Berdasarkan Kamus Leksikon)", fontweight="bold")
                plt.xlabel("Prediksi (Naïve Bayes)", fontweight="bold")
                st.pyplot(fig_cm)
                plt.close(fig_cm)