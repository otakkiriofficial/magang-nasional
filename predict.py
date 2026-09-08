import os
import pickle
from scipy.sparse import hstack
from preprocessing import clean_text
from utils import hitung_skor_leksikon

MODEL_DIR = 'models'
_cache = {}

def _load_artifacts():
    """Memuat model Naive Bayes dan objek pendukung (Vectorizer, Lexicon) ke dalam cache."""
    if _cache: 
        return _cache
    
    files = [
        ('tfidf_vectorizer.pkl', 'vec'), 
        ('lexicon_scaler.pkl', 'scaler'), 
        ('naive_bayes_model.pkl', 'model'), 
        ('lexicon_dict.pkl', 'lex')
    ]
    
    for fname, key in files:
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Berkas {fname} tidak ditemukan. Silakan jalankan menu 'Pelatihan Data' terlebih dahulu.")
        with open(path, 'rb') as f:
            _cache[key] = pickle.load(f)
    return _cache

def predict_sentiment(raw_text: str) -> dict:
    """Melakukan inferensi sentimen pada teks opini tunggal dari media sosial."""
    arts = _load_artifacts()
    
    # 1. Pra-pemrosesan teks (Pembersihan, Case Folding, Stopwords, dll)
    teks_bersih = clean_text(raw_text)
    if not teks_bersih:
        # Jika teks kosong setelah dibersihkan, kembalikan probabilitas seimbang (fallback)
        return {'label': 'netral', 'skor_leksikon': 0.0, 'proba': {'negatif': 0.33, 'netral': 0.34, 'positif': 0.33}}
    
    # 2. Vektorisasi TF-IDF
    tfidf_vec = arts['vec'].transform([teks_bersih])
    
    # 3. Hitung dan sesuaikan rentang skor leksikon
    skor_mentah = hitung_skor_leksikon(teks_bersih, arts['lex'])
    skor_scaled = arts['scaler'].transform([[skor_mentah]])
    
    # 4. Penggabungan Vektor (TF-IDF + Lexicon Features)
    X_input = hstack([tfidf_vec, skor_scaled])
    
    # 5. Klasifikasi dengan Naive Bayes
    label = arts['model'].predict(X_input)[0]
    proba_array = arts['model'].predict_proba(X_input)[0]
    
    # Menggabungkan nama kelas dengan nilai probabilitasnya
    proba_dict = dict(zip(arts['model'].classes_, proba_array))
    
    return {
        'label': label,
        'skor_leksikon': skor_mentah,
        'proba': {k: round(float(v), 3) for k, v in proba_dict.items()}
    }

if __name__ == '__main__':
    # Pengetesan skrip internal untuk memastikan pipeline prediksi berjalan dengan baik
    # Disimulasikan dengan opini terkait Kereta Cepat Whoosh
    kalimat = "kereta cepat whoosh memang luar biasa canggih dan nyaman, tapi sayang utangnya sangat membebani apbn negara"
    hasil = predict_sentiment(kalimat)
    
    print("="*50)
    print("SISTEM PREDIKSI SENTIMEN | NAIVE BAYES")
    print("="*50)
    print(f"Kalimat Uji       : '{kalimat}'")
    print(f"Hasil Klasifikasi : {hasil['label'].upper()}")
    print(f"Skor Leksikon     : {hasil['skor_leksikon']}")
    print(f"Probabilitas      : {hasil['proba']}")
    print("="*50)