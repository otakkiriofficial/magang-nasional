import os
import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from utils import KATA_NEGASI

# ==========================================
# 1. KONFIGURASI REGEX PEMBERSIHAN
# Dioptimalkan untuk karakteristik teks Media Sosial (Facebook)
# ==========================================
URL_PATTERN = re.compile(r'https?://\S+|www\S+')
MENTION_PATTERN = re.compile(r'@[A-Za-z0-9_]+')
HASHTAG_PATTERN = re.compile(r'#\S+') 
HTML_ENTITIES = re.compile(r'&[a-z]+;') # Membersihkan sisa elemen web seperti &amp;, &quot;
NON_ALPHA_PATTERN = re.compile(r'[^a-z\s]')
EXTRA_SPACES_PATTERN = re.compile(r'\s+')

# ==========================================
# 2. PEMUATAN KAMUS NORMALISASI
# ==========================================
kamus_normalisasi = {}
try:
    # Memuat kamus singkatan
    path_singkatan = os.path.join('data', 'singkatan-lib.csv')
    if os.path.exists(path_singkatan):
        df_singkatan = pd.read_csv(path_singkatan, header=None, names=['slang', 'baku'])
        kamus_normalisasi.update({str(k).strip().lower(): str(v).strip().lower() for k, v in zip(df_singkatan['slang'], df_singkatan['baku'])})
    
    # Memuat kamus alay/bahasa gaul
    path_alay = os.path.join('data', 'colloquial-indonesian-lexicon.csv')
    if os.path.exists(path_alay):
        df_alay = pd.read_csv(path_alay, usecols=['slang', 'formal'])
        kamus_normalisasi.update({str(k).strip().lower(): str(v).strip().lower() for k, v in zip(df_alay['slang'], df_alay['formal'])})
except Exception as e:
    print(f"Catatan: Gagal memuat sebagian kamus normalisasi: {e}")

# ==========================================
# 3. PEMUATAN STOPWORDS
# ==========================================
stopwords_set = set()
try:
    path_stopwords = os.path.join('data', 'indonesia-stopwords.txt')
    if os.path.exists(path_stopwords):
        with open(path_stopwords, 'r', encoding='utf-8') as f:
            stopwords_set = {line.strip().lower() for line in f.readlines() if line.strip()}
        
        # Mengecualikan kata negasi dari daftar stopwords agar konteks sentimen tidak hilang
        stopwords_set = stopwords_set - KATA_NEGASI
except Exception as e:
    print(f"Catatan: Gagal memuat berkas stopwords: {e}")

# ==========================================
# 4. INISIALISASI SASTRAWI STEMMER
# ==========================================
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    """Pipeline pembersihan teks secara menyeluruh untuk setiap kalimat opini."""
    if pd.isna(text):
        return ""
    
    # Case folding (mengecilkan huruf)
    text = str(text).lower()
    
    # Cleansing pola khusus
    text = URL_PATTERN.sub(' ', text)
    text = MENTION_PATTERN.sub(' ', text)
    text = HASHTAG_PATTERN.sub(' ', text)
    text = HTML_ENTITIES.sub(' ', text)
    
    # Membuang angka, tanda baca, dan emoji
    text = NON_ALPHA_PATTERN.sub(' ', text)
    
    # Membuang spasi berlebih
    text = EXTRA_SPACES_PATTERN.sub(' ', text).strip()
    
    # Tokenisasi manual berbasis spasi
    words = text.split()
    if not words:
        return ""
    
    cleaned_words = []
    for w in words:
        # Normalisasi: Ubah kata gaul/singkatan menjadi kata baku
        w_baku = kamus_normalisasi.get(w, w)
        # Filtering: Hapus stopword (dengan mempertahankan kata negasi)
        if w_baku not in stopwords_set:
            cleaned_words.append(w_baku)
            
    # Menggabungkan kembali token menjadi kalimat dan melakukan proses Stemming
    return stemmer.stem(" ".join(cleaned_words))

def run_preprocessing_pipeline(df, text_column='textDisplay'):
    """
    Fungsi utama untuk diaplikasikan ke dalam Pandas DataFrame.
    text_column: Sesuaikan dengan nama kolom yang menampung ulasan di dataset Facebook Anda.
    """
    df = df.copy()
    
    # Terapkan fungsi pembersihan ke seluruh baris data
    df['clean_text'] = df[text_column].apply(clean_text)
    
    # Eliminasi baris yang secara tidak sengaja kosong setelah proses pembersihan
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)
    
    return df

if __name__ == '__main__':
    # Blok Pengetesan Skrip Internal
    contoh_teks = "Wahhh kereta WHOOSH cepet bgt!! 🚄🔥 Tp tiketnya mhl bgt min @KAI #Whoosh &amp; bkin utang numpuk 😭 www.beritautama.com"
    
    print("="*50)
    print("PENGUJIAN MODUL PRA-PEMROSESAN (PREPROCESSING)")
    print("="*50)
    print(f"Teks Asli   : {contoh_teks}")
    print(f"Hasil Clean : {clean_text(contoh_teks)}")
    print("="*50)