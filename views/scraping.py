import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
import re

def get_video_id(url):
    """Fungsi pembantu untuk mengekstrak ID Video dari URL YouTube"""
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def scrape_youtube_comments(api_key, video_id, max_results):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None

    while len(comments) < max_results:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response['items']:
            snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                "author": snippet['authorDisplayName'],
                "textDisplay": snippet['textDisplay'],
                "likeCount": snippet['likeCount'],
                "publishedAt": snippet['publishedAt']
            })

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
            
    return pd.DataFrame(comments)

def show_scraping_page():
    # Header bergaya profesional
    st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #1B4965; margin-bottom: 25px">
            <h2 style="color: #1B4965; margin: 0;">📥 YouTube Data Acquisition</h2>
            <p style="color: #62B6CB; margin: 0;">Koleksi opini publik menggunakan YouTube Data API v3</p>
            <p style="color: #62B6CB; margin: 0;">(YouTube Data API harus daftar terlebih dahulu di Google Cloud Console</p>
            
        </div>
    """, unsafe_allow_html=True)

    # Area Konfigurasi API
    with st.expander("🔑 Konfigurasi API & Kredensial", expanded=True):
        api_key = st.text_input("YouTube API Key", type="password", help="Dapatkan di Google Cloud Console")
        col_url, col_limit = st.columns([3, 1])
        with col_url:
            video_url = st.text_input("URL Video YouTube", placeholder="https://www.youtube.com/watch?v=...")
        with col_limit:
            limit = st.number_input("Limit Komentar", min_value=10, max_value=5000, value=100)

    if st.button("🚀 Mulai Scraping Komentar", type="primary", use_container_width=True):
        if not api_key or not video_url:
            st.warning("⚠️ Mohon lengkapi API Key dan URL Video.")
            return

        video_id = get_video_id(video_url)
        if not video_id:
            st.error("❌ URL YouTube tidak valid.")
            return

        try:
            with st.status("📡 Menghubungkan ke Server Google API...", expanded=True) as status:
                st.write(f"Menarik data untuk ID Video: `{video_id}`")
                df = scrape_youtube_comments(api_key, video_id, limit)
                
                if not df.empty:
                    status.update(label=f"✅ Berhasil mengambil {len(df)} komentar!", state="complete")
                    
                    # Ringkasan hasil dalam Metric Cards
                    m1, m2 = st.columns(2)
                    m1.metric("Komentar Didapat", len(df))
                    m2.metric("Video ID", video_id)

                    st.divider()
                    st.write("**Preview Data Mentah:**")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Tombol Download
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Dataset (CSV)",
                        data=csv,
                        file_name=f"yt_comments_{video_id}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.info("💡 File ini dapat langsung diunggah ke menu 'Analisis & Laporan'.")
                else:
                    status.update(label="⚠️ Tidak ada komentar ditemukan.", state="error")

        except Exception as e:
            st.error(f"Terjadi kesalahan API: {str(e)}")