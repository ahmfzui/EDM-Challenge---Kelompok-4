import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import joblib
import pandas as pd
import re
from keras.models import load_model
from indoNLP.preprocessing import replace_slang
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from indoNLP.preprocessing import *
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bullying Detection", layout="wide")

# Load models and vectorizer
svm = joblib.load('models/svm_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Preprocessing functions
def convert_emoji(text):
    return emoji_to_words(text, lang='id', use_alias=False, delimiter=(' ', ' '))

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)  # Remove mentions
    text = re.sub(r'#\S+', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)  # Keep only alphanumeric
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'(.)\1+', r'\1', text)  # Remove duplicate characters
    return text

def stopword(str_text):
    stop_words = StopWordRemoverFactory().get_stop_words()
    new_array = ArrayDictionary(stop_words)
    stop_words_remover_new = StopWordRemover(new_array)
    return stop_words_remover_new.remove(str_text)

def tokenize_text(text):
    return text.split()

def stemming(text_cleaning):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return ' '.join([stemmer.stem(word) for word in text_cleaning])

from scipy.special import expit  # Sigmoid function

def predict_hate_speech(text, model):
    # Preprocessing steps
    text = convert_emoji(text)
    text = clean_text(text)
    text = text.lower()
    text = replace_slang(text)
    text = stopword(text)
    text = tokenize_text(text)
    text = stemming(text)
    
    # Count the word distribution and filter
    word_count = len(text.split())
    if word_count == 0 or word_count > 25:
        return None  # Skip this text if it doesn't meet the criteria
    
    # Transform text using the vectorizer
    text_transformed = vectorizer.transform([text])
    
    # Predicting the class
    prediction = model.predict(text_transformed)
    
    result = 'Bullying' if prediction == 1 else 'Non-Bullying'
    
    return result


def main():
    menu_selection = option_menu(
        menu_title=None,
        options=["Home", "Bullying Detection"],
        icons=["house-fill", "exclamation-triangle"],
        default_index=0,
        orientation="horizontal",
    )

    if menu_selection == "Home":
        # Home content
        st.markdown("""
        <style>
            .title {
                text-align: center;
                font-size: 36px;
                font-weight: bold;
            }
            .header {
                text-align: center;
                font-size: 24px;
                font-weight: bold;
            }
            table {
                margin-top: 50px;
                margin-bottom: 30px;
                width: 80%;
                text-align: center;
                border-collapse: collapse;
            }
            th, td {
                padding: 10px;
                border: 1px solid #ddd;
            }
            th {
                background-color: #E4A800;  /* Updated background color */
                color: #fff;  /* Updated text color */
            }
            img {
                max-width: 800px;
                width: 100%;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div align="center">
            <h1><strong>EDM Challenge - Kelompok 4</strong></h1>
            <table>
                <tr>
                    <th>Nama Anggota</th>
                    <th>NIM</th>
                </tr>
                <tr>
                    <td>Ahmad Fauzi</td>
                    <td>1202220263</td>
                </tr>
                <tr>
                    <td>Alvaro Cleosanda</td>
                    <td>1202220181</td>
                </tr>
                <tr>
                    <td>Vilson</td>
                    <td>1202220199</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---", unsafe_allow_html=True)
        st.markdown('<div style="text-align: center"><h1>🎯 <strong>Sentinel: Deteksi Komentar Bullying di Media Sosial</strong></h1></div>', unsafe_allow_html=True)
        
        # Load and display the image
        col1, col2, col3 = st.columns([0.2, 0.7, 0.2])
        with col2:
            image = Image.open('img/artificial (2000 x 800 piksel) (1).png')
            st.image(image, use_column_width=True)
        
        st.write("""
        Sentinel adalah project dari kelompok kami untuk menghasilkan sebuah model analisis sentimen yang dirancang untuk mendeteksi komentar di media sosial, khususnya untuk mengidentifikasi apakah sebuah komentar termasuk dalam kategori bullying atau non-bullying. Proyek ini menggabungkan model machine learning dengan implementasi dalam bentuk website berbasis Streamlit dan notebook, bertujuan untuk mengurangi cyberbullying di sosial media.
        Model machine learning yang diusulkan akan dikembangkan untuk mengidentifikasi dan mengklasifikasikan komentar di media sosial ke dalam dua kategori: **Bullying** dan **Non-Bullying**. Dengan menggunakan dataset yang telah diberi label dari komentar TikTok, model ini akan dilatih untuk mengenali pola, bahasa, dan konteks yang membedakan antara komentar yang merugikan dan yang tidak.
        """)
        
        st.markdown("#### 📊 **Overview Dataset**", unsafe_allow_html=True)
        st.write("""
        Dataset yang digunakan dalam proyek ini diperoleh melalui proses scraping data komentar di TikTok, menggunakan script dari repository GitHub [cubernetes/TikTokCommentScraper](https://github.com/cubernetes/TikTokCommentScraper). File utama yang digunakan dalam proyek ini adalah `train.csv`, yang berisi data komentar TikTok beserta label klasifikasinya. Berikut adalah deskripsi dari kolom-kolom yang ada dalam dataset tersebut:

        - **Text**: Kolom ini berisi teks dari komentar yang diambil dari video TikTok. Setiap entri adalah komentar unik dari pengguna TikTok.
        - **Label**: Kolom ini berisi label untuk setiap komentar yang menunjukkan apakah komentar tersebut termasuk dalam kategori **Bullying** atau **Non-Bullying**. Label ini berisi nilai biner:
          - **0**: Non-Bullying, menunjukkan bahwa komentar tersebut tidak mengandung bullying.
          - **1**: Bullying, menunjukkan bahwa komentar tersebut mengandung bullying.
        """)

        st.markdown("<br>", unsafe_allow_html=True)

        # Load and display CSV data
        df = pd.read_csv('Data Scrapping - EDM Challenge Kelompok 4 - Sheet1.csv')
        st.dataframe(df, use_container_width=True)
        
        st.markdown("---", unsafe_allow_html=True)
    
    elif menu_selection == "Bullying Detection":
        tab1, tab2 = st.tabs(["Input Text 📝", "Upload Dataset CSV 📂"])
        st.markdown("""
            <style>
            .stButton { display: flex; justify-content: center; }
            </style>
            """, unsafe_allow_html=True)      

        with tab1:
            col1, col2, col3 = st.columns([0.2, 0.8, 0.2])  # Adjust column widths
            with col2:  # Center column
                st.markdown("<br>", unsafe_allow_html=True)
                with st.form(key='bullying_detection_form'):
                    user_input = st.text_area("Input Text", placeholder="Masukkan teks komentar di sini...", height=200)
                    st.markdown("<br>", unsafe_allow_html=True)  # Add space above the button
                    submit_button = st.form_submit_button("Submit")  # Ensure label is not empty
                    st.markdown("<br>", unsafe_allow_html=True)  # Add space below the button
                    
                    if submit_button:
                        # Call your prediction function
                        result = predict_hate_speech(user_input, svm)
                        
                        # Display the result
                        if result == "Bullying":
                            col1, col2, col3 = st.columns([0.2, 0.295, 0.2])
                            with col2:
                                st.warning(f"Teks tersebut teridentifikasi sebagai **Bullying** 🤨")
                        else:
                            col1, col2, col3 = st.columns([0.2, 0.3405, 0.2])
                            with col2:
                                st.success(f"Teks tersebut teridentifikasi sebagai **Non-Bullying** 😊")

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form(key='upload_csv_form'):
                uploaded_file = st.file_uploader("Unggah file CSV untuk dideteksi", type="csv")
                st.markdown("<br>", unsafe_allow_html=True)
                upload_button = st.form_submit_button("Upload")
                st.markdown("<br>", unsafe_allow_html=True)
                
                if upload_button and uploaded_file is not None:
                    try:
                        # Membaca file CSV dengan encoding yang sesuai
                        df_upload = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

                        if 'Text' in df_upload.columns:
                            # Menggunakan apply untuk mendapatkan prediksi
                            predictions = df_upload['Text'].apply(lambda x: predict_hate_speech(x, svm))
                            
                            # Menambahkan hasil prediksi ke dataframe
                            df_upload['Prediction'] = [result for result in predictions]
                            df_upload = df_upload.dropna()

                            col1, col2= st.columns([0.5, 0.5])
                            with col1:
                                # Menampilkan dataframe hasil prediksi
                                st.dataframe(df_upload, use_container_width=True, height=695)
                            with col2:
                                # Membuat pie chart dari hasil prediksi
                                prediction_counts = df_upload['Prediction'].value_counts()
                                
                                # Menentukan warna khusus untuk Non-Bullying (hijau) dan Bullying (#E4A800)
                                colors = ['#379C5E', '#E4A800']

                                # Menambahkan efek explode pada kedua label untuk memperjelas tampilan pie chart
                                explode = [0.03, 0.03]  # Sedikit memisahkan kedua bagian pie chart

                                # Menyiapkan label dan data untuk pie chart
                                labels = [f'Non-Bullying ({prediction_counts.get(0, 0)})', f'Bullying ({prediction_counts.get(1, 0)})']
                                sizes = prediction_counts
                                total = sum(sizes)
                                
                                def autopct_func(pct, allvalues):
                                    absolute = int(round(pct / 100.*total))
                                    return f'{pct:.1f}% ({absolute})'

                                # Membuat pie chart dengan ukuran yang lebih kecil (4x4)
                                plt.figure(figsize=(5,5))
                                wedges, texts, autotexts = plt.pie(sizes, 
                                                                autopct=lambda pct: autopct_func(pct, sizes), 
                                                                colors=colors, 
                                                                startangle=140, 
                                                                explode=explode, 
                                                                wedgeprops={'edgecolor': 'black'}, 
                                                                textprops={'fontsize': 8})  # Kecilkan font keterangan

                                # Menambahkan judul pada pie chart dengan font yang lebih kecil
                                plt.title('Distribusi Prediksi Bullying dan Non-Bullying', fontsize=10)

                                # Menambahkan legenda
                                plt.legend(wedges, labels, loc="lower right", fontsize=7)
                                
                                # Menampilkan pie chart di Streamlit
                                st.pyplot(plt, bbox_inches="tight")
                        
                        else:
                            st.error("File CSV harus memiliki kolom 'Text'.")
                    
                    except pd.errors.EmptyDataError:
                        st.error("File CSV kosong atau tidak dapat dibaca.")
                    except pd.errors.ParserError:
                        st.error("Terjadi kesalahan saat mem-parsing file CSV.")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses file: {str(e)}")

if __name__ == "__main__":
    main()
