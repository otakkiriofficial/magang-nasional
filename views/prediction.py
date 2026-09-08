import pandas as pd
import numpy as np
import streamlit as st
from predict import predict_sentiment, _load_artifacts
from preprocessing import run_preprocessing_pipeline
from utils import KATA_NEGASI

def show_prediction_page(model=None, vectorizer=None):
    # Header halaman yang selaras dengan tema utama aplikasi
    st.markdown(
        """
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #1B4965; margin-bottom: 25px">
            <h2 style="color: #1B4965; margin: 0;">⚡ Prediksi Kata Cepat</h2>
            <p style="color: #62B6CB; margin: 0;">Simulasi pengujian sentimen teks tunggal dan transparansi perhitungan algoritma (White-Box Testing)</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⌨️ Input Komentar Pengujian")
    input_text = st.text_area(
        "Teks Opini Publik / Komentar:",
        height=120,
        placeholder="Contoh: Program magang nasional 2025 ini sangat bagus untuk meningkatkan kompetensi mahasiswa sebelum lulus, tapi kuota pendaftarannya terlalu sedikit dan seleksinya ketat.",
    )

    if st.button("🚀 Mulai Analisis Sentimen", type="primary", use_container_width=True):
        if input_text.strip():
            st.info("🔍 **Memulai proses ekstraksi...** Sistem sedang menjalankan seluruh tahapan pipeline Natural Language Processing (NLP).")
            
            with st.spinner("⚙️ Menghitung probabilitas..."):
                
                # --- PROSES PREPROCESSING ---
                df_temp = pd.DataFrame({"textDisplay": [input_text]})
                df_clean = run_preprocessing_pipeline(df_temp, text_column="textDisplay")

                if df_clean.empty or str(df_clean.iloc[0]["clean_text"]).strip() == "":
                    st.warning("⚠️ **Teks Invalid:** Teks tidak mengandung kata bermakna atau hanya berisi simbol/spam setelah melewati proses pembersihan.")
                else:
                    clean_text = df_clean.iloc[0]["clean_text"]

                    try:
                        hasil_prediksi = predict_sentiment(clean_text)
                        arts = _load_artifacts() 
                    except FileNotFoundError as e:
                        st.error(f"⚠️ **Sistem Peringatan:** Komponen berkas model belum lengkap ({e}).")
                        return

                    # Hasil Naive Bayes
                    pred_label = hasil_prediksi["label"].capitalize()
                    prob_dict = hasil_prediksi["proba"]
                    prob_max = max(prob_dict.values())
                    
                    # Hasil Leksikon
                    raw_score = hasil_prediksi["skor_leksikon"]
                    if raw_score > 0:
                        label_leksikon = "Positif"
                    elif raw_score < 0:
                        label_leksikon = "Negatif"
                    else:
                        label_leksikon = "Netral"

                    st.success("✅ **Kalkulasi Selesai.** Integrasi pembobotan TF-IDF dan skor semantik leksikon berhasil dipetakan.")

                    # --- TAMPILAN KEPUTUSAN UTAMA ---
                    st.divider()
                    st.markdown("### 📊 Hasil Keputusan Akhir Algoritma")

                    if pred_label == "Positif":
                        st.success(f"#### Klasifikasi Akhir: POSITIF (Confidence Score: {prob_max:.2%})")
                    elif pred_label == "Negatif":
                        st.error(f"#### Klasifikasi Akhir: NEGATIF (Confidence Score: {prob_max:.2%})")
                    else:
                        st.warning(f"#### Klasifikasi Akhir: NETRAL (Confidence Score: {prob_max:.2%})")

                    # --- TAMPILAN RINCIAN PERHITUNGAN ---
                    with st.expander("🔍 Buka Rincian Perhitungan Matematis Lengkap", expanded=True):
                        
                        # TAHAP 1: PREPROCESSING
                        st.markdown("#### 1. Text Processing (Pembersihan Teks)")
                        st.write("Transformasi teks mentah menjadi kata bersih melalui tahapan case folding, tokenizing, normalisasi kata slang, filtering stopword, dan stemming Sastrawi.")
                        st.code(f"Kalimat Sebelum : {input_text}\nKalimat Sesudah : {clean_text}", language="text")

                        # TAHAP 2: TF-IDF
                        st.markdown("#### 2. Perhitungan TF-IDF")
                        st.write("Bobot statistik untuk setiap kata unik yang relevan di dalam dokumen berdasarkan korpus pelatihan:")
                        tfidf_vec = arts['vec'].transform([clean_text])
                        feature_names = arts['vec'].get_feature_names_out()
                        nonzero_indices = tfidf_vec.nonzero()[1]
                        
                        if len(nonzero_indices) > 0:
                            tfidf_data = {"Kata Fitur": [], "Bobot TF-IDF": []}
                            for idx in nonzero_indices:
                                tfidf_data["Kata Fitur"].append(feature_names[idx])
                                tfidf_data["Bobot TF-IDF"].append(round(tfidf_vec[0, idx], 5))
                            st.table(pd.DataFrame(tfidf_data).sort_values(by="Bobot TF-IDF", ascending=False))
                        else:
                            st.write("*(Tidak ada kata kunci yang cocok dengan kosakata TF-IDF model)*")

                        # TAHAP 3: LEKSIKON
                        st.markdown("#### 3. Perhitungan Kamus Leksikon")
                        st.write("Pencocokan nilai polaritas semantik kata berdasarkan kamus sentimen TSV (mendukung penanganan kata negasi secara bertahap):")
                        lexicon_dict = arts['lex']
                        words = clean_text.split()
                        negasi_aktif = False
                        lex_data = {"Kata": [], "Status": [], "Skor": []}
                        
                        for w in words:
                            if w in KATA_NEGASI:
                                negasi_aktif = True
                                lex_data["Kata"].append(w)
                                lex_data["Status"].append("Kata Negasi (Membalik polaritas kata berikutnya)")
                                lex_data["Skor"].append(0.0)
                            elif w in lexicon_dict:
                                bobot = lexicon_dict[w]
                                if negasi_aktif:
                                    skor_balik = bobot * -1
                                    lex_data["Kata"].append(w)
                                    lex_data["Status"].append(f"Dibalik karena Negasi (Asli: {bobot})")
                                    lex_data["Skor"].append(skor_balik)
                                    negasi_aktif = False
                                else:
                                    lex_data["Kata"].append(w)
                                    lex_data["Status"].append("Teridenteksi di Kamus")
                                    lex_data["Skor"].append(bobot)
                        
                        if lex_data["Kata"]:
                            st.table(pd.DataFrame(lex_data))
                        else:
                            st.write("*(Tidak ada kata opini yang terdaftar pada Kamus Leksikon)*")
                            
                        st.markdown(f"**Total Skor Leksikon Mentah:** `{raw_score}` ➡️ **(Label Awal Leksikon: {label_leksikon.upper()})**")

                        # TAHAP 4: NORMALISASI
                        st.markdown("#### 4. Normalisasi Fitur (MinMaxScaler)")
                        st.write("Mengubah rentang nilai skor leksikon menjadi format non-negatif agar kompatibel dengan input Multinomial Naïve Bayes.")
                        skor_scaled = arts['scaler'].transform([[raw_score]])[0][0]
                        st.markdown(f"**Hasil Penskalaan Leksikon:** `{skor_scaled:.5f}`")

                        # TAHAP 5: NAIVE BAYES
                        st.markdown("#### 5. Teorema Klasifikasi Multinomial Naïve Bayes")
                        st.write("Komparasi nilai probabilitas posterior akhir dari gabungan fitur statistik dan nilai semantik untuk penentuan kelas:")
                        for kelas, prob in prob_dict.items():
                            if kelas.lower() == pred_label.lower():
                                st.success(f"**$P(\\text{{{kelas.capitalize()}}}|\\text{{Teks}})$** = {prob:.4f} ({prob:.2%}) ⬅️ **Probabilitas Tertinggi (Keputusan Final)**")
                            else:
                                st.info(f"**$P(\\text{{{kelas.capitalize()}}}|\\text{{Teks}})$** = {prob:.4f} ({prob:.2%})")
                    
                    # ==========================================
                    # --- RINGKASAN PERUBAHAN KEPUTUSAN ---
                    # ==========================================
                    st.markdown("---")
                    st.markdown("### 📝 Ringkasan Analisis Gabungan")
                    
                    st.info(f"**1. Kalimat Pengujian (Mentah):**\n{input_text}")
                    
                    if label_leksikon == pred_label:
                        st.success(
                            f"**2. Hasil Sentimen Leksikon (Kamus):** {label_leksikon.upper()}\n\n"
                            f"**3. Hasil Akhir Naïve Bayes:** {pred_label.upper()}\n\n"
                            f"*(Algoritma Naïve Bayes memperkuat dan menyetujui klasifikasi berbasis Kamus Leksikon)*"
                        )
                    else:
                        st.warning(
                            f"**2. Hasil Sentimen Leksikon (Kamus):** {label_leksikon.upper()}\n\n"
                            f"**3. Hasil Akhir Naïve Bayes:** {pred_label.upper()}\n\n"
                            f"*(Algoritma Naïve Bayes melakukan koreksi terhadap keputusan Kamus Leksikon, karena berdasarkan distribusi bobot TF-IDF pada data latih, pola tekstual kalimat ini mengarah kuat pada kelas {pred_label.upper()})*"
                        )
                            
        else:
            st.warning("Kolom input komentar tidak boleh kosong!")