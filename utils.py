import os
import pandas as pd
import streamlit as st
import pickle  

# ==========================================
# 1. KONFIGURASI KATA NEGASI 
# ==========================================
KATA_NEGASI = frozenset({
    'tidak', 'tak', 'bukan', 'jangan', 'belum', 'kurang',
    'enggak', 'gak', 'ga', 'nggak', 'tanpa', 'tiada',
    'anti', 'gagal', 'hampir', 'nyaris', 'sulit', 'susah'
})

# ==========================================
# 2. PEMUATAN LEKSIKON (TSV DENGAN HEADER WORD & WEIGHT)
# ==========================================
@st.cache_data
def get_combined_lexicon():
    """Memuat dan menggabungkan leksikon positif & negatif dari fail berformat TSV."""
    def load_tsv_lexicon(path):
        if not os.path.exists(path):
            return {}
        try:
            # Membaca file TSV secara eksplisit
            df = pd.read_csv(path, sep='\t', on_bad_lines='skip')
            
            # Memeriksa apakah file menggunakan nama kolom standar (word & weight)
            if 'word' in df.columns and 'weight' in df.columns:
                words = df['word'].astype(str).str.lower().str.strip()
                weights = pd.to_numeric(df['weight'], errors='coerce').fillna(0)
            else:
                # Fallback: Jika tidak terdeteksi, baca ulang tanpa header lalu eliminasi manual
                df = pd.read_csv(path, sep='\t', header=None, on_bad_lines='skip')
                if str(df.iloc[0, 0]).strip().lower() in ['word', 'kata']:
                    df = df.iloc[1:].reset_index(drop=True)
                words = df.iloc[:, 0].astype(str).str.lower().str.strip()
                weights = pd.to_numeric(df.iloc[:, 1], errors='coerce').fillna(0)
                
            return dict(zip(words, weights))
        except Exception as e:
            print(f"Peringatan: Gagal memuat leksikon dari {path}. Error: {e}")
            return {}

    pos_lex = load_tsv_lexicon(os.path.join('data', 'positive.tsv'))
    neg_lex = load_tsv_lexicon(os.path.join('data', 'negative.tsv'))
    
    # Gabungkan kamus, ambil bobot absolut tertinggi jika terjadi bentrokan/duplikasi kata
    combined_lexicon = {**pos_lex}
    for word, score in neg_lex.items():
        if word not in combined_lexicon or abs(score) > abs(combined_lexicon[word]):
            combined_lexicon[word] = score
            
    return combined_lexicon

# ==========================================
# 3. ALGORITMA PERHITUNGAN SKOR SENTIMEN
# ==========================================
def hitung_skor_leksikon(text, lexicon_dict):
    """Logika iteratif untuk menghitung total skor leksikon dari 1 kalimat opini."""
    words = str(text).split()
    score = 0.0
    negasi_aktif = False
    
    for word in words:
        if word in KATA_NEGASI:
            negasi_aktif = True
            continue
        
        if word in lexicon_dict:
            bobot = lexicon_dict[word]
            # Jika kata didahului kata negasi, balikkan polaritas skornya (+ menjadi -, dan sebaliknya)
            score += bobot * (-1 if negasi_aktif else 1)
            negasi_aktif = False  # Reset efek negasi untuk kata berikutnya
            
    return score

# ==========================================
# 4. PEMUATAN MODEL AI (SINKRONISASI DENGAN TRAIN_MODEL)
# ==========================================
@st.cache_resource(show_spinner="Memuat Model AI...")
def load_models():
    """
    Memuat model Naive Bayes dan TF-IDF Vectorizer dari folder 'models'.
    Path ini sudah disesuaikan dengan output dari file train_model.py.
    """
    # PERBAIKAN PATH: Mengarahkan pembacaan file ke dalam folder 'models'
    model_path = os.path.join('models', 'naive_bayes_model.pkl')
    vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None
        
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
            
        return model, vectorizer
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

if __name__ == '__main__':
    # Blok Pengetesan Skrip Internal
    dummy_lexicon = {'bermanfaat': 4, 'sulit': -3, 'sukses': 4, 'kecewa': -4}
    
    # Pengujian efek negasi pada konteks opini
    contoh_teks = "program magang ini tidak sulit dan sangat bermanfaat"
    skor = hitung_skor_leksikon(contoh_teks, dummy_lexicon)
    
    print("="*50)
    print("PENGUJIAN MODUL UTILS (LEKSIKON & NEGASI)")
    print("="*50)
    print(f"Kalimat     : '{contoh_teks}'")
    print(f"Kamus Aktif : {dummy_lexicon}")
    print(f"Skor Final  : {skor}")
    print("="*50)