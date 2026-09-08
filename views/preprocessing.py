import streamlit as st
import pandas as pd

def show_preprocessing_page():
    st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #1B4965; margin-bottom: 25px">
            <h2 style="color: #1B4965; margin: 0;">🧹 Pembersihan Relevancy</h2>
            <p style="color: #62B6CB; margin: 0;">Penyaringan dataset mentah untuk memastikan relevansi dengan konteks Program Magang Nasional 2025</p>
        </div>
    """, unsafe_allow_html=True)

    # 1. Fitur Upload Dataset
    st.markdown("### 📂 Input Dataset Mentah")
    uploaded_file = st.file_uploader("Pilih file CSV dataset hasil Scraping", type=["csv"])

    if uploaded_file is not None:
        # Membaca data
        df = pd.read_csv(uploaded_file)
        
        # Validasi apakah kolom target ada
        if 'textDisplay' in df.columns:
            df_clean = df[['textDisplay']].dropna().rename(columns={'textDisplay': 'komentar'})
        elif 'komentar' in df.columns:
            df_clean = df[['komentar']].dropna()
        else:
            st.error("❌ Dataset harus memiliki kolom bernama 'textDisplay' atau 'komentar'.")
            return

        with st.expander("👁️ Lihat Sampel Data Mentah", expanded=False):
            st.dataframe(df_clean.head(10), use_container_width=True)

        # 2. Kamus Kata Kunci (Dibuat interaktif menggunakan text_area agar bisa diedit langsung di UI)
        st.markdown("### ⚙️ Konfigurasi Kamus Pemfilteran")
        col1, col2 = st.columns(2)
        
        with col1:
            # Menggunakan kata kunci spesifik Magang Nasional 2025
            default_akurat = "magang, nasional, program, magang nasional, magang 2025, peserta, mahasiswa, kampus, kampus merdeka, msib, internship, pendaftaran, seleksi, rekrutmen, pelatihan, sertifikat, mentor, perusahaan, kementerian, kemendikbud, skill, karir, pengalaman kerja, fresh graduate, uang saku, gaji, insentif, pengangguran, beban kerja, eksploitasi, kritik, program pemerintah, lowongan, kerja, anak magang, digital, administrasi, kompetensi, produktivitas, evaluasi, peserta magang"
            input_akurat = st.text_area("Kamus Konten Relevan (pisahkan dengan koma)", default_akurat, height=250)
            kamus_konten_akurat = [k.strip().lower() for k in input_akurat.split(',')]

        with col2:
            # Menggunakan kata kunci spam
            default_spam = "promo, cek profil, kuota gratis, jualan, klik disini, shopee, call"
            input_spam = st.text_area("Kamus Spam (pisahkan dengan koma)", default_spam, height=250)
            kamus_spam = [k.strip().lower() for k in input_spam.split(',')]

        # 3. Fungsi Filtering
        def filter_otomatis_akurat(teks):
            teks_lower = str(teks).lower()

            # Aturan 1: Jika terdeteksi spam -> TIDAK_VALID
            if any(spam in teks_lower for spam in kamus_spam):
                return "TIDAK_VALID"

            # Aturan 2: Jika terlalu pendek (kurang dari 3 kata) -> TIDAK_VALID
            if len(teks_lower.split()) < 3:
                return "TIDAK_VALID"

            # Aturan 3: Cek relevansi konten terhadap kamus kata kunci
            ada_konteks = any(keyword in teks_lower for keyword in kamus_konten_akurat)
            if ada_konteks:
                return "VALID"
            else:
                return "TIDAK_VALID"

        # 4. Tombol Eksekusi
        if st.button("🚀 Mulai Proses Filtering", type="primary", use_container_width=True):
            with st.spinner('Memindai dan memfilter ribuan baris komentar...'):
                df_clean['status_validasi'] = df_clean['komentar'].apply(filter_otomatis_akurat)
                
                # Memisahkan data valid untuk diekspor
                df_valid = df_clean[df_clean['status_validasi'] == 'VALID'].copy()
                
                st.success("✅ Proses penyaringan relevansi selesai!")
                
                # Menampilkan ringkasan visual
                st.markdown("### 📊 Ringkasan Status Penyaringan")
                
                c_met1, c_met2, c_met3 = st.columns(3)
                c_met1.metric("Total Data Awal", len(df_clean))
                c_met2.metric("Data Valid (Relevan)", len(df_valid))
                c_met3.metric("Data Tidak Valid (Dibuang)", len(df_clean) - len(df_valid))

                st.bar_chart(df_clean['status_validasi'].value_counts())

                # 5. Fitur Download Data Hasil Terfilter
                # Data yang diunduh hanya yang memiliki status 'VALID'
                df_export = df_valid[['komentar']]
                csv = df_export.to_csv(index=False, encoding='utf-8')
                
                st.markdown("### 📥 Ekspor Dataset")
                st.info("💡 Unduh dataset yang sudah bersih ini. File inilah yang akan Anda gunakan pada halaman **Training Model** selanjutnya.")
                
                st.download_button(
                    label="💾 Unduh Dataset Valid (Format CSV)",
                    data=csv,
                    file_name='dataset-magang-nasional-terfilter.csv',
                    mime='text/csv',
                    use_container_width=True
                )