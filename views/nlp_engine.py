import os
import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Mengambil KATA_NEGASI untuk mengamankan stopwords dari penghapusan
try:
    from utils import KATA_NEGASI
except ImportError:
    KATA_NEGASI = set()

# ==========================================
# 1. KONFIGURASI REGEX PEMBERSIHAN
# Dioptimalkan untuk data Media Sosial (Facebook)
# ==========================================
URL_PATTERN = re.compile(r'https?://\S+|www\S+')
MENTION_PATTERN = re.compile(r'@[A-Za-z0-9_]+')
HASHTAG_PATTERN = re.compile(r'#\S+')
HTML_ENTITIES = re.compile(r'&[a-z]+;')
NON_ALPHA_PATTERN = re.compile(r'[^a-z\s]')
EXTRA_SPACES_PATTERN = re.compile(r'\s+')

# ==========================================
# 2. PEMUATAN KAMUS NORMALISASI
# ==========================================
kamus_normalisasi = {}
try:
    path_singkatan = os.path.join('data', 'singkatan-lib.csv')
    if os.path.exists(path_singkatan):
        df_singkatan = pd.read_csv(path_singkatan, header=None, names=['slang', 'baku'])
        kamus_normalisasi.update({str(k).strip().lower(): str(v).strip().lower() for k, v in zip(df_singkatan['slang'], df_singkatan['baku'])})
    
    path_alay = os.path.join('data', 'colloquial-indonesian-lexicon.csv')
    if os.path.exists(path_alay):
        df_alay = pd.read_csv(path_alay, usecols=['slang', 'formal'])
        kamus_normalisasi.update({str(k).strip().lower(): str(v).strip().lower() for k, v in zip(df_alay['slang'], df_alay['formal'])})
except Exception:
    pass

# ==========================================
# 3. PEMUATAN STOPWORDS
# ==========================================
stopwords_set = set()
try:
    path_stopwords = os.path.join('data', 'indonesia-stopwords.txt')
    if os.path.exists(path_stopwords):
        with open(path_stopwords, 'r', encoding='utf-8') as f:
            stopwords_set = {line.strip().lower() for line in f.readlines() if line.strip()}
        # Mengecualikan kata negasi dari set stopwords
        stopwords_set = set(stopwords_set) - set(KATA_NEGASI)
except Exception:
    pass

# ==========================================
# 4. INISIALISASI STEMMER & CACHE
# ==========================================
stemmer = StemmerFactory().create_stemmer()
stem_cache = {}

# ==========================================
# FUNGSI TAHAPAN PREPROCESSING TERPISAH
# ==========================================

def case_folding(text):
    if pd.isna(text): return ""
    return str(text).lower()

def cleaning(text):
    text = URL_PATTERN.sub(' ', text)
    text = MENTION_PATTERN.sub(' ', text)
    text = HASHTAG_PATTERN.sub(' ', text)
    text = HTML_ENTITIES.sub(' ', text)
    text = NON_ALPHA_PATTERN.sub(' ', text)
    return EXTRA_SPACES_PATTERN.sub(' ', text).strip()

def tokenizing(text):
    return text.split()

def normalizing(tokens):
    return [kamus_normalisasi.get(w, w) for w in tokens]

def filtering(tokens):
    return [w for w in tokens if w not in stopwords_set]

def stemming(tokens):
    stemmed_tokens = []
    for word in tokens:
        if word not in stem_cache:
            stem_cache[word] = stemmer.stem(word)
        stemmed_tokens.append(stem_cache[word])
    return stemmed_tokens

# ==========================================
# PIPELINE UTAMA UNTUK DATAFRAME
# ==========================================

def run_preprocessing_pipeline(df, text_column='textDisplay'):
    """
    Fungsi ini akan menghasilkan kolom-kolom terpisah untuk setiap tahapan.
    Sangat berguna untuk diekspor ke Excel sebagai lampiran skripsi.
    """
    df_processed = df.copy()
    
    df_processed['case_folding'] = df_processed[text_column].apply(case_folding)
    df_processed['cleaning']     = df_processed['case_folding'].apply(cleaning)
    df_processed['tokenizing']   = df_processed['cleaning'].apply(tokenizing)
    df_processed['normalizing']  = df_processed['tokenizing'].apply(normalizing)
    df_processed['filtering']    = df_processed['normalizing'].apply(filtering)
    df_processed['stemming']     = df_processed['filtering'].apply(stemming)
    
    # Menggabungkan token akhir kembali menjadi kalimat utuh untuk proses TF-IDF
    df_processed['clean_text']   = df_processed['stemming'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
    
    # Hapus baris yang kosong setelah diproses
    df_processed = df_processed[df_processed['clean_text'].str.strip() != ''].reset_index(drop=True)
    
    return df_processed

if __name__ == '__main__':
    # Blok Pengetesan Skrip Internal disesuaikan dengan topik Whoosh
    contoh_teks = "Wahhh kereta WHOOSH ini cepet bgt!! 🔥 Tp tiketnya mhl &amp; bkin utang negara numpuk 😭 www.beritautama.com"
    
    print("="*50)
    print("PENGUJIAN MODUL NLP ENGINE TERPISAH")
    print("="*50)
    print("Teks Asli   :", contoh_teks)
    print("-" * 50)
    
    cf = case_folding(contoh_teks)
    cl = cleaning(cf)
    tk = tokenizing(cl)
    nm = normalizing(tk)
    fl = filtering(nm)
    st_res = stemming(fl)
    
    print("1. Case Fold   :", cf)
    print("2. Cleaning    :", cl)
    print("3. Tokenizing  :", tk)
    print("4. Normalizing :", nm)
    print("5. Filtering   :", fl)
    print("6. Stemming    :", " ".join(st_res))
    print("="*50)