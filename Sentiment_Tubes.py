#!/usr/bin/env python
# coding: utf-8

# ### **Pemodelan Topik LDA dan Analisis Sentimen BERT Untuk Optimalisasi Kebijakan Green Economy dan Energi Baru Terbarukan (EBT) di Indonesia: Studi Media Sosial X**

#  **1. Amanda Putri Aprilliani (105222001)**
#  
#  **2. Raihan Akira Rahmaputra (105222040)**
# 
#  **3. Gema Fitri Ramadani (105222009)**
# 
#  **4. Anom Wajawening (105222029)**

# > 

# ----------

# ### **Scraping Tweet Social Media X** 

# Kode ini mengimplementasikan proses pengumpulan data tweet menggunakan Twitter API v2 dengan library Tweepy untuk mengekstrak diskusi publik terkait Energi Baru Terbarukan (EBT) di Indonesia. 
# 
# Query pencarian menggunakan kombinasi kata kunci berbahasa Indonesia dan Inggris seperti "energi terbarukan", "solar panel", "ESDM", dan "climate change" dengan operator Boolean untuk memastikan relevansi data, serta memfilter tweet berbahasa Indonesia sambil mengecualikan retweet, reply, dan hashtag politik tertentu untuk menghindari bias.
# 
# Proses scraping dibatasi maksimal 1.000 tweet dengan pagination menggunakan next_token, dan setiap request diberi jeda 1 detik untuk menghindari rate limiting. Data yang dikumpulkan meliputi User ID, timestamp pembuatan tweet, teks tweet, dan URL tweet yang kemudian disimpan dalam format CSV dengan encoding UTF-8.
# 
# Pendekatan ini memungkinkan pengumpulan data real-time yang representatif untuk analisis sentimen dan pemodelan topik dalam konteks kebijakan energi terbarukan Indonesia.

# In[ ]:


import tweepy
import pandas as pd
import time

# Ganti dengan Bearer token Anda
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAPhRxQEAAAAA3kbv2Jt6CIhlU07vAb8s4n3Vur4%3D34yjU0FdbFPHc3kOEzlLTz5gcP26IsAhwNM8RjhPnAr6frIR5R'

# Menggunakan Bearer token untuk otentikasi
client = tweepy.Client(bearer_token=bearer_token)

# Define search query
search_query = (
    '("renewable energy" OR "energi terbarukan" OR "EBT" OR "NRE" OR'
    '"energi baru" OR "solar panel" OR "panel surya" OR "wind power" OR'
    '"geothermal" OR "biomass" OR'
    '"ESDM" OR "Bahlil Lahadalia" OR "energi hidro" OR'
    '"PLTP" OR "energi bersih" OR "climate change" OR "deforestasi" OR'
    '"transisi energi" OR "energi hijau" OR'
    '"keberlanjutan energi")'
    '-is:retweet -is:reply (lang:id)'
    '-#SwasembadaEnergi -#PresidenPrabowo -#KetahananPangan -#AstaCita -#TransisiEnergi -#SobatNKRI'  # Contoh exclude hashtags
) # Ubah kata kunci yang mau diambil, gunakan logika AND dan OR dan pastikan data sudah tepat diambil
max_results = 100  # Maksimum 100 per permintaan
total_tweets = 0
attributes_container = []
next_token = None

try:
    while total_tweets < 1000:  # Batasi dulu maks 2000 tweet
        # Mengambil tweet menggunakan API v2
        response = client.search_recent_tweets(query=search_query, max_results=max_results, tweet_fields=['created_at', 'author_id'], next_token=next_token)

        # Memeriksa apakah ada tweet yang ditemukan
        if response.data:
            for tweet in response.data:
                tweet_url = f"https://twitter.com/twitter/status/{tweet.id}"
                attributes_container.append([tweet.author_id, tweet.created_at, tweet.text, tweet_url])
                total_tweets += 1  

        
        next_token = response.meta.get('next_token')
        if not next_token:
            break  

        # Menambahkan jeda untuk menghindari rate limit
        time.sleep(1)  # Jeda 1 detik antara permintaan

    
    columns = ["User   ID", "Date Created", "Tweet Text", "Tweet URL"] #tambahkan data geo
    tweets_df = pd.DataFrame(attributes_container, columns=columns)

    # Menampilkan DataFrame
    print(tweets_df)

    # Menyimpan DataFrame ke CSV dengan encoding UTF-8
    tweets_df.to_csv('hasil-scraping.csv', index=False, encoding='utf-8') 

except Exception as e:
    print('Status Failed On,', str(e))


# ### **EDA**

# In[1]:


import pandas as pd
df = pd.read_csv('hasil-scraping.csv', encoding='utf-8')


# In[2]:


df.info()


# In[3]:


df.head()


# >---

# ### **Data Preprocessing**

# In[4]:


import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# **Kamus Slang Bahasa Indonesia**

# Kamus ini didapatkan dari dataset yang dikirimkan ke LLM (Claude Sonnet 4) untuk dianalisis kira-kira apa saja slang dictionary yang dibutuhkan/disesuaikan dalam dataset project ini. 

# In[5]:


# Indonesian slang dictionary - optimized untuk EBT context
INDONESIAN_SLANG_DICT = {
    # Basic slang normalization
    ' ga ': ' tidak ',
    ' gak ': ' tidak ',
    ' gk ': ' tidak ',
    ' yg ': ' yang ',
    ' udh ': ' sudah ',
    ' udah ': ' sudah ',
    ' bgt ': ' banget ',
    ' gmn ': ' bagaimana ',
    ' dmn ': ' dimana ',
    ' krn ': ' karena ',
    ' karna ': ' karena ',
    ' tp ': ' tapi ',
    ' hrs ': ' harus ',
    ' emg ': ' memang ',
    ' emang ': ' memang ',
    ' org ': ' orang ',
    ' dr ': ' dari ',
    ' dgn ': ' dengan ',
    ' utk ': ' untuk ',
    ' bs ': ' bisa ',
    ' jd ': ' jadi ',
    ' nggak ': ' tidak ',
    ' ngga ': ' tidak ',
    ' aja ': ' saja ',
    ' cuma ': ' hanya ',
    ' dah ': ' sudah ',
    ' tau ': ' tahu ',
    ' kalo ': ' kalau ',
    ' abis ': ' habis ',
    ' nih ': ' ini ',
    ' tuh ': ' itu ',
    ' yah ': ' ya ',
    ' kan ': ' kan ',
    ' kok ': ' kok ',
    # Energy/EBT specific
    ' pln ': ' PLN ',
    ' esdm ': ' ESDM ',
    ' ebt ': ' EBT ',
    ' plts ': ' PLTS ',
    ' pltp ': ' PLTP ',
    ' plta ': ' PLTA '
}

print(f"Slang dictionary created with {len(INDONESIAN_SLANG_DICT)} entries")


# **Hapus Tweet Duplikat**

# In[6]:


def remove_duplicates(df, text_column='Tweet Text'):
    """Remove duplicate tweets"""
    print(f"Before removing duplicates: {len(df)} tweets")
    df_clean = df.drop_duplicates(subset=[text_column], keep='first')
    print(f"After removing duplicates: {len(df_clean)} tweets")
    print(f"Removed {len(df) - len(df_clean)} duplicate tweets")
    return df_clean

# Apply duplicate removal
df_step1 = remove_duplicates(df)


# **Mengganti url dengan [url]**

# In[7]:


def clean_urls(text):
    text = re.sub(r'http\S+|https\S+|www\S+', '', text)  # Remove completely
    text = re.sub(r'\[URL\]', '', text)  # Remove [URL] tokens
    return text  # Return cleaned textcleaning

test_text = "Lihat artikel ini https://example.com tentang energi terbarukan"
print(f"Original: {test_text}")
print(f"Cleaned:  {clean_urls(test_text)}")


# **Mengganti mention user dengan [USER]**

# In[8]:


def clean_mentions(text):
    """Remove user mentions"""
    text = re.sub(r'@\w+', '[USER]', text)
    return text

# Test mention cleaning
test_text = "@jokowi Panel surya bagus untuk @indonesia"
print(f"Original: {test_text}")
print(f"Cleaned:  {clean_mentions(test_text)}")


# **Menghapus hastag**

# In[9]:


def clean_hashtags(text):
    """Clean hashtags - keep the text, remove the # symbol"""
    text = re.sub(r'#(\w+)', r'\1', text)
    return text

# Test hashtag cleaning
test_text = "Dukung #EnergiTerbarukan dan #PanelSurya untuk masa depan"
print(f"Original: {test_text}")
print(f"Cleaned:  {clean_hashtags(test_text)}")


# **Menghapus tanda baca berlebihan**

# In[10]:


def remove_excessive_punctuation(text):
    """Remove excessive punctuation while keeping meaning"""
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[,]{2,}', ',', text)
    text = re.sub(r'[;]{2,}', ';', text)
    text = re.sub(r'[-]{3,}', '-', text)
    return text

# Test punctuation cleaning
test_text = "Wah bagus banget!!! Kok bisa??? Mantap..."
print(f"Original: {test_text}")
print(f"Cleaned:  {remove_excessive_punctuation(test_text)}")


# **Normalisasi spasi yang berlebih**

# In[11]:


def normalize_whitespace(text):
    """Normalize whitespace and remove extra spaces"""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Test whitespace normalization
test_text = "Panel   surya    sangat     bagus    untuk    masa   depan"
print(f"Original: '{test_text}'")
print(f"Cleaned:  '{normalize_whitespace(test_text)}'")


# **Normalisasi slang dengan bahasa indonesia**

# In[12]:


def normalize_slang(text, slang_dict=INDONESIAN_SLANG_DICT):
    """Normalize Indonesian slang to formal words"""
    text_lower = text.lower()
    text_spaced = ' ' + text_lower + ' '
    
    for slang, formal in slang_dict.items():
        text_spaced = text_spaced.replace(slang.lower(), formal.lower())
    
    text_normalized = text_spaced.strip()
    return text_normalized

# Test slang normalization
test_text = "Panel surya yg udh dipasang ga mahal bgt kok"
print(f"Original: {test_text}")
print(f"Cleaned:  {normalize_slang(test_text)}")


# **Menjalankan preprocessing**

# In[13]:


def preprocess_text(text, keep_emojis=True):
    """
    Combined preprocessing function
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Apply all cleaning steps
    text = clean_urls(text)
    text = clean_mentions(text)
    text = clean_hashtags(text)
    
    
    text = remove_excessive_punctuation(text)
    text = normalize_whitespace(text)
    text = normalize_slang(text)
    text = normalize_whitespace(text)
    
    return text

# Test complete preprocessing
test_text = "Wah @jokowi panel surya yg udh dipasang bagus bgt!!! 😊 https://example.com #EBT"
print(f"Original:  {test_text}")
print(f"Processed: {preprocess_text(test_text)}")


# In[14]:


# Apply preprocessing to all tweets
print("Applying preprocessing to all tweets...")
df_step1['Tweet Text_processed'] = df_step1['Tweet Text'].apply(
    lambda x: preprocess_text(x, keep_emojis=True)
)

print("Preprocessing completed!")
print(f"Dataset shape: {df_step1.shape}")


# In[15]:


# Show before/after examples
print("="*80)
print("PREPROCESSING EXAMPLES:")
print("="*80)

n_examples = 5
for i in range(min(n_examples, len(df_step1))):
    original = df_step1.iloc[i]['Tweet Text']
    processed = df_step1.iloc[i]['Tweet Text_processed']

    print(f"\nExample {i+1}:")
    print("-" * 50)
    print(f"ORIGINAL:  {original}")
    print(f"PROCESSED: {processed}")


# **Kata-kata yang sering muncul**

# In[16]:


# Analyze most common words after preprocessing
all_words = []
for text in df_step1['Tweet Text_processed']:
    if pd.notna(text):
        all_words.extend(text.split())

word_counts = Counter(all_words)
print("Top 20 most common words after preprocessing:")
for word, count in word_counts.most_common(20):
    print(f"{word}: {count}")


# **Processed text disimpan dalam hasilscraping-processed.csv**

# In[17]:


# Save processed dataset
output_filename = 'hasilscraping-processed.csv'
df_step1.to_csv(output_filename, index=False, encoding='utf-8')

print(f"Processed data saved to: {output_filename}")
print(f"Final dataset info:")
print(f"- Total tweets: {len(df_step1)}")
print(f"- Columns: {list(df_step1.columns)}")
print(f"- Ready for LDA, feature engineering and model training!")


# > -----

# ### **Topic Modelling dengan LDA**

# **Import Library**

# In[18]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pyLDAvis


# **Load dataset hasilscraping-processed.csv**

# In[19]:


# Load preprocessed data
df = pd.read_csv('hasilscraping-processed.csv')

print(f"Preprocessed data loaded!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Use the processed text column
text_column = 'Tweet Text_processed'
texts = df[text_column].dropna().tolist()

print(f"\nTotal tweets for topic modeling: {len(texts)}")
print(f"Sample processed tweets:")
for i in range(3):
    print(f"{i+1}. {texts[i][:100]}...")


# **Preprocessed untuk LDA, Mengubah data teks menjadi format numerik (Document-Term Matrix)**

# In[20]:


# Prepare text data for LDA
def prepare_text_for_lda(texts, max_features=800, min_df=3, max_df=0.8):
    """
    Prepare text for LDA using CountVectorizer
    """
    # Use CountVectorizer for LDA
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=min_df,       # Ignore terms that appear in less than min_df documents
        max_df=max_df,       # Ignore terms that appear in more than max_df of documents
        stop_words=None,     # We already cleaned in preprocessing
        lowercase=True,
        token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic tokens, min 2 chars
    )
    
    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"Document-term matrix shape: {doc_term_matrix.shape}")
    print(f"Vocabulary size: {len(feature_names)}")
    
    return doc_term_matrix, vectorizer, feature_names

# Create document-term matrix
doc_term_matrix, vectorizer, feature_names = prepare_text_for_lda(texts)

# Show top terms by frequency
term_freq = doc_term_matrix.sum(axis=0).A1
term_freq_df = pd.DataFrame({
    'term': feature_names,
    'frequency': term_freq
}).sort_values('frequency', ascending=False)

print("\nTop 20 most frequent terms:")
print(term_freq_df.head(20))


# **Menentukan jumlah topik yang optimal untuk model LDA**
# 1. Metrik Perplexity (Keheranan) : mengukur seberapa baik model yang sudah dilatih dapat memprediksi data baru (data yang tidak terlihat). Nilai yang lebih rendah lebih baik. Skala perplexity dari 1 hingga tak terhingga.
# 2. Coherence (Koherensi/kerelevanan) : mengukur sebarapa masuk akal topik yang dihasilkan bagi manusia. Nilai yang lebih tinggi lebih baik.
# 

# In[21]:


# Set style seaborn untuk tampilan bersih
sns.set_style("whitegrid")

# Buat plot dengan warna yang profesional
plt.figure(figsize=(12, 6))

def compute_coherence_scores(doc_term_matrix, vectorizer, topic_range):
    """
    Compute coherence scores for different numbers of topics
    """
    coherence_scores = []
    perplexity_scores = []
    
    for n_topics in topic_range:
        print(f"Testing {n_topics} topics...")
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100,
            learning_method='batch',
            n_jobs=-1
        )
        
        lda.fit(doc_term_matrix)
        
        # Calculate perplexity (lower is better)
        perplexity = lda.perplexity(doc_term_matrix)
        perplexity_scores.append(perplexity)
        
        coherence_scores.append(-perplexity) 
    
    return coherence_scores, perplexity_scores

# Test different numbers of topics
topic_range = range(5, 13, 2)  
print("Computing coherence scores for different topic numbers...")
coherence_scores, perplexity_scores = compute_coherence_scores(doc_term_matrix, vectorizer, topic_range)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot coherence scores
ax1.plot(topic_range, coherence_scores, 'bo-')
ax1.set_xlabel('Number of Topics')
ax1.set_ylabel('Coherence Score')
ax1.set_title('Coherence Score vs Number of Topics')
ax1.grid(True)

# Plot perplexity scores
ax2.plot(topic_range, perplexity_scores, 'ro-')
ax2.set_xlabel('Number of Topics')
ax2.set_ylabel('Perplexity')
ax2.set_title('Perplexity vs Number of Topics')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Find optimal number of topics
optimal_idx = np.argmax(coherence_scores)
optimal_topics = list(topic_range)[optimal_idx]
print(f"\nOptimal number of topics: {optimal_topics}")


# 1. Grafik coherence score (kiri) : Grafik ini menunjukkan skor tertinggi (puncak) saat jumlah topik adalah 11. Hal ini mengindikasikan bahwa dengan 11 topik, kata-kata dalam setiap topik paling relevan dan topik-topiknya paling mudah dibedakan maknanya.
# 2. Grafik Perplexity (kanan) : Nilai perplexity terus menurun dan mencapai titik terendahnya pada 11 topik. Hal ini mendukung bahwa model lebih "percaya diri" dan tidak "bingung" saat memodelkan data dengan 11 topik.

# **Training LDA**

# In[22]:


# Train LDA with optimal number of topics
def train_lda_model(doc_term_matrix, n_topics, random_state=42):
    """
    Train final LDA model
    """
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        max_iter=100,
        learning_method='batch',
        doc_topic_prior=None,  # Use 1/n_topics
        topic_word_prior=None, # Use 1/n_topics
        n_jobs=-1
    )
    
    print(f"Training LDA model with {n_topics} topics...")
    lda_model.fit(doc_term_matrix)
    
    # Calculate final scores
    perplexity = lda_model.perplexity(doc_term_matrix)
    log_likelihood = lda_model.score(doc_term_matrix)
    
    print(f"Model training completed!")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Log-likelihood: {log_likelihood:.4f}")
    
    return lda_model

# Train final model
final_lda = train_lda_model(doc_term_matrix, optimal_topics)


# 1. Perplexity: 170.3105
# 
# Ini adalah skor "kebingungan" final dari model LDA. Angka ini tidak dinilai "baik" atau "buruk" secara tunggal, namun angka inilah skor terbaik (terendah) yang dicapai selama proses pengujian untuk memilih jumlah topik.
# 
# 2. Log-likelihood: -146668.8655
# 
# Ini adalah skor yang mengukur seberapa baik model "cocok" (fit) dengan data. Semakin tinggi nilainya (semakin dekat ke nol), semakin baik. Angka ini mewakili kecocokan terbaik yang bisa dicapai model dengan 11 topik.

# **Menampilkan topik dari hasil LDA**

# In[23]:


def display_topics(lda_model, feature_names, n_top_words=10):
    """
    Display topics with their top words
    """
    topics = []
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = [topic[i] for i in top_words_idx]
        
        topics.append({
            'topic_id': topic_idx,
            'words': top_words,
            'weights': top_weights
        })
        
        print(f"\nTopic {topic_idx}:")
        print(f"Top words: {', '.join(top_words[:8])}")
        
        # Create word-weight pairs
        word_weights = [(word, weight) for word, weight in zip(top_words, top_weights)]
        for word, weight in word_weights[:5]:
            print(f"  {word}: {weight:.4f}")
    
    return topics

# Display topics
topics = display_topics(final_lda, feature_names, n_top_words=15)


# #### **Analisis Core Themes**
# 
# 1. ENERGY_TECHNOLOGY (Topics 1, 6)
# * Fokus pada aspek teknologi energi terbarukan
# * Mencakup diskusi tentang energi baru dan transisi energi di Indonesia
# * Melibatkan pembicaraan tentang PLN dan infrastruktur energi
# 
# 
# 2. ENVIRONMENTAL_IMPACT (Topics 2, 8, 10)
# * Fokus pada dampak lingkungan, terutama deforestasi dan kelapa sawit
# * Mencakup perdebatan tentang keberlanjutan lingkungan
# * Melibatkan isu-isu kontroversial terkait sawit dan penggundulan hutan
# 
# 
# 3. GOVERNMENT_POLICY (Topics 7, 9)
# * Fokus pada kebijakan pemerintah dan lembaga terkait
# * Mencakup diskusi tentang ESDM, menteri, dan tokoh politik (Bahlil Lahadalia)
# * Melibatkan aspek regulasi dan administrasi energi
# 
# 4. ENERGY_ACCESS (Topics 3, 5)
# * Fokus pada akses energi untuk masyarakat
# * Mencakup diskusi tentang gas bumi untuk rakyat dan kebijakan terkait
# * Melibatkan aspek keadilan sosial dan pemerataan energi
# 
# 
# 5. OTHER (Topics 0, 4)
# * Kategori untuk topik-topik umum atau noise
# * Mencakup kata-kata umum atau topik yang kurang spesifik

# ### **Kategorisasi Topik**

# **Import Library**

# In[24]:


import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess


# In[25]:


# Set style seaborn untuk tampilan bersih
sns.set_style("whitegrid")

# Buat plot dengan warna yang profesional
plt.figure(figsize=(12, 6))


# **Menerapkan core theme/topik utama LDA ke dalam data**

# In[26]:


# Langkah 1: Memetakan topik dominan ke dalam dokumen
def get_dominant_topic(lda_model, doc_term_matrix, texts):
    # Transformasi dokumen ke topik
    topic_distributions = lda_model.transform(doc_term_matrix)
    
    dominant_topics = []
    for i, topic_dist in enumerate(topic_distributions):
        # Mendapatkan topik dengan probabilitas tertinggi
        dominant_topic = topic_dist.argmax()
        dominant_topics.append({
            'Document_Id': i,
            'Dominant_Topic': dominant_topic,
            'Topic_Prob': topic_dist[dominant_topic],
            'Text': texts[i]
        })
    return dominant_topics

# Mendapatkan topik dominan untuk setiap dokumen
dominant_topics = get_dominant_topic(final_lda, doc_term_matrix, df['Tweet Text_processed'].tolist())

# Membuat DataFrame hasil
results = pd.DataFrame(dominant_topics)

# Langkah 2: Tambahkan kategori ke DataFrame hasil topik
def assign_core_theme(topic_idx):
    core_themes = {
        'ENERGY_TECHNOLOGY': [1, 6],
        'ENVIRONMENTAL_IMPACT': [2, 8, 10],
        'GOVERNMENT_POLICY': [7, 9],
        'ENERGY_ACCESS': [3, 5],
        'OTHER': [0, 4]
    }
    
    for theme, topics in core_themes.items():
        if topic_idx in topics:
            return theme
    return "UNDEFINED"

# Terapkan ke DataFrame hasil LDA
results['Core_Theme'] = results['Dominant_Topic'].apply(assign_core_theme)

# Analisis distribusi core themes
theme_distribution = results['Core_Theme'].value_counts()
print("Distribusi Core Themes:")
print(theme_distribution)

# Visualisasi distribusi core themes
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
theme_distribution.plot(kind='bar', color='skyblue')
plt.title('Distribusi Core Themes dalam Dataset')
plt.xlabel('Core Theme')
plt.ylabel('Jumlah Tweet')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('core_themes_distribution.png')
plt.show()

# Gabungkan hasil dengan DataFrame asli
df_with_topics = pd.merge(df, results[['Document_Id', 'Dominant_Topic', 'Core_Theme']], 
                         left_index=True, right_on='Document_Id', how='left')

# Simpan hasil
df_with_topics.to_csv('hasil_dengan_topik.csv', index=False)


# In[27]:


core_themes = {
    'ENERGY_TECHNOLOGY': [1, 6],
    'ENVIRONMENTAL_IMPACT': [2, 8, 10],
    'GOVERNMENT_POLICY': [7, 9],
    'ENERGY_ACCESS': [3, 5],
    'OTHER': [0, 4]
}


# **Distribusi topik per core themes**

# In[28]:


# Analisis distribusi topik per core theme
print("\nDistribusi Topik per Core Theme:")
for theme in core_themes.keys():
    theme_docs = results[results['Core_Theme'] == theme]
    print(f"\n{theme}:")
    print(theme_docs['Dominant_Topic'].value_counts())


# ### **Evaluasi Model LDA Menggunakan Metrik C-v dan UMass**

# In[29]:


import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

texts = [doc.split() for doc in df['Tweet Text_processed']]
dictionary = Dictionary(texts)

topics = []
for topic_idx, topic in enumerate(final_lda.components_):
    top_words_idx = topic.argsort()[-10:][::-1]  # 10 kata teratas
    top_words = [feature_names[i] for i in top_words_idx]
    topics.append(top_words)

# Hitung coherence score dengan metode yang berbeda
coherence_cv = CoherenceModel(topics=topics, texts=texts, 
                             dictionary=dictionary, coherence='c_v')
c_v = coherence_cv.get_coherence()

coherence_umass = CoherenceModel(topics=topics, texts=texts, 
                                dictionary=dictionary, coherence='u_mass')
c_umass = coherence_umass.get_coherence()

print(f"Coherence Score (C_v): {c_v:.4f}")
print(f"Coherence Score (UMass): {c_umass:.4f}")


# * Coherence Skor (C_v) : 0.6576. Skor ini tergolong tinggi, mengonfirmasi bahwa topik-topik yang dihasilkan sangat koheren dan kata-kata di dalamnya berhubungan secara semantik (makna).
# * Coherence Skor (UMass) : -3.3650. Untuk UMass, nilai yang lebih dekat ke nol dianggap lebih baik. Hasil ini memberikan validasi tambahan dari sudut pandang metrik yang berbeda.

# **Visualisasi wordcloud**

# In[30]:


# Visualisasi wordcloud per core theme
from wordcloud import WordCloud

for theme in core_themes.keys():
    theme_docs = results[results['Core_Theme'] == theme]
    theme_texts = ' '.join(theme_docs['Text'].tolist())
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(theme_texts)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud - {theme}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'wordcloud_{theme}.png')
    plt.show()


# ### **Feature Engineering**

# **Preprocessing sebelum masuk ke labeling**

# In[31]:


def prepare_for_bert(text):
    # Hapus karakter khusus yang mungkin mengganggu tokenisasi
    text = re.sub(r'[^\w\s]', ' ', text)
    # Hapus multiple spaces
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
    
df_sentiment = df_with_topics.copy()
df_sentiment['Text_for_BERT'] = df_sentiment['Tweet Text_processed'].apply(prepare_for_bert)


# In[32]:


from transformers import AutoTokenizer

# Load tokenizer IndoBERT
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

def encode_text(texts, max_length=128):
    """Encode teks untuk IndoBERT"""
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

# Contoh encoding untuk beberapa teks
sample_texts = df_sentiment['Text_for_BERT'].iloc[:5].tolist()
encoded = encode_text(sample_texts)
print("Bentuk input_ids:", encoded['input_ids'].shape)
print("Bentuk attention_mask:", encoded['attention_mask'].shape)


# In[33]:


df_sentiment


# In[34]:


# Encoding teks untuk seluruh dataset
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, pipeline

# 1. Load pre-trained IndoBERT model untuk encoding
indobert_model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")

# Fungsi untuk mendapatkan embeddings dari IndoBERT
def get_bert_embeddings(texts, max_length=128, batch_size=16):
    """Mendapatkan embeddings dari IndoBERT model"""
    all_embeddings = []
    
    # Proses dalam batch untuk menghemat memori
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = encode_text(batch_texts, max_length)
        
        # Mendapatkan embeddings dari model
        with torch.no_grad():
            outputs = indobert_model(**inputs)
            # Mengambil [CLS] embedding (representasi keseluruhan kalimat)
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings)
    
    # Menggabungkan semua batch
    if len(all_embeddings) > 0:
        return torch.cat(all_embeddings, dim=0)
    else:
        return None

# Contoh penggunaan untuk subset kecil
small_sample = df_sentiment['Text_for_BERT'].iloc[:10].tolist()
embeddings = get_bert_embeddings(small_sample)
print(f"Embedding shape: {embeddings.shape}")  # Seharusnya [10, 768]


# In[35]:


df = pd.read_csv('hasilscraping-processed.csv')

df.info()


# >---

# ### **Sentiment Labelling using LLM**

# **Preprocessing data sebelum melakukan labeling. Yang dilakukan adalah penghapusan duplikat, merapikan kembali urutan indeks setelah beberapa baris dihapus, drop kolom Tweet_Text (tweet asli), dan menghasilkan file output 'ready_labelling.csv'**

# In[36]:


import pandas as pd

# Baca file CSV
df = pd.read_csv('hasilscraping-processed.csv')  

# Lihat nama kolom yang sebenarnya untuk memastikan
print("Nama kolom dalam dataset:")
print(df.columns.tolist())

# Cetak informasi awal
print(f"Jumlah total tweets sebelum menghapus duplikat: {len(df)}")
print(f"Jumlah unique values di Tweet Text_processed: {df['Tweet Text_processed'].nunique()}")

# Identifikasi duplikat
duplicates = df[df.duplicated(subset=['Tweet Text_processed'], keep='first')]
print(f"Jumlah duplikat ditemukan: {len(duplicates)}")

# Hapus duplikat berdasarkan kolom 'Tweet Text_processed'
df_unique = df.drop_duplicates(subset=['Tweet Text_processed'], keep='first')

# Reset index
df_unique = df_unique.reset_index(drop=True)

# Cetak informasi setelah menghapus duplikat
print(f"Jumlah tweets setelah menghapus duplikat: {len(df_unique)}")

# Drop kolom 'Tweet Text' jika ada
if 'Tweet Text' in df_unique.columns:
    df_unique = df_unique.drop(columns=['Tweet Text'])
    print("Kolom 'Tweet Text' telah dihapus")
else:
    print("Kolom 'Tweet Text' tidak ditemukan dalam dataset")

# Simpan dataset tanpa duplikat ke file CSV baru
output_file = 'ready_labelling.csv'
df_unique.to_csv(output_file, index=False)
print(f"Dataset tanpa duplikat telah disimpan ke {output_file}")


if len(duplicates) > 0:
    print("\nContoh tweet duplikat yang dihapus:")

    user_id_column = None
    possible_user_id_columns = ['User ID', 'User   ID', 'UserID', 'User_ID']
    
    for col in possible_user_id_columns:
        if col in df.columns:
            user_id_column = col
            break
    
    if user_id_column:
        for i, row in duplicates.head(3).iterrows():
            print(f"Tweet: {row['Tweet Text_processed']}")
            
            # Cari tweet asli (first occurrence) yang dipertahankan
            original = df[df['Tweet Text_processed'] == row['Tweet Text_processed']].iloc[0]
            print(f"Original tweet ID: {original[user_id_column]}, Duplicate tweet ID: {row[user_id_column]}")
            print("---")
    else:
        # Jika tidak ada kolom User ID yang cocok, tampilkan tanpa ID
        for i, row in duplicates.head(3).iterrows():
            print(f"Tweet: {row['Tweet Text_processed']}")
            print("---")


#  **Data yang siap untuk dilabeli melalui LLM**

# In[38]:


df= pd.read_csv('ready_labelling.csv')
df.info()


# > **Telah dilakukan labelling dengan bantuan LLM. Model yang digunakan adalah Claude Sonnet 4**

# **Data yang sudah dilabeli LLM**

# In[ ]:


import pandas as pd

# Solusi simple dan efektif
data_label = pd.read_csv('labeling_claude.csv', sep=';')

if data_label.shape[1] > 5:  
    data_label = data_label.iloc[:, :-1]  # Drop kolom terakhir

print("✅ Berhasil!")
print(f"Shape: {data_label.shape}")
print(data_label.head())


# In[40]:


data_label.tail(5)


# **EDA Data Label LLM**

# In[41]:


# 2. Target distribution analysis
print("\n🎯 SENTIMENT DISTRIBUTION:")
sentimen_dist = data_label['Sentimen'].value_counts()
for sent, count in sentimen_dist.items():
    pct = (count / len(data_label)) * 100
    print(f"  {sent}: {count} ({pct:.1f}%)")


# In[42]:


sns.set_style("white")
sns.set_context("paper", font_scale=1.2)

# Create figure with larger size
plt.figure(figsize=(12, 6))

# Create bar plot using seaborn
sns.countplot(data=data_label, x='Sentimen', palette='RdYlBu')

# Customize plot
plt.title('Distribusi Sentimen dalam Dataset', pad=15)
plt.xlabel('Kategori Sentimen', labelpad=10)
plt.ylabel('Jumlah Tweet', labelpad=10)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add value labels on top of each bar
for i in plt.gca().containers:
    plt.gca().bar_label(i, padding=3)

# Adjust layout
plt.tight_layout()

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# Show plot
plt.show()


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")
sns.set_context("paper", font_scale=1.2)

# Create figure with larger size
plt.figure(figsize=(12, 6))

# Create bar plot using seaborn
sns.countplot(data=data_label, x='Aspek', palette='RdYlBu')

# Customize plot
plt.title('Distribusi Aspek dalam Dataset', pad=15)
plt.xlabel('Kategori Aspek', labelpad=10)
plt.ylabel('Jumlah Tweet', labelpad=10)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add value labels on top of each bar
for i in plt.gca().containers:
    plt.gca().bar_label(i, padding=3)

# Adjust layout
plt.tight_layout()

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7, axis='y')

# Show plot
plt.show()


# Berdasarkan grafik, Kebijakan Pemerintah (GOVERNMENT_POLICY) secara jelas menjadi tema yang paling mendominasi diskusi publik tentang EBT di media sosial X. Topik mengenai Teknologi Energi (ENERGY_TECHNOLOGY) juga merupakan sorotan utama, sementara isu Akses Energi (ENERGY_ACCESS) adalah yang paling jarang dibicarakan. Hal ini mengindikasikan bahwa diskursus publik lebih terfokus pada aspek regulasi dan inovasi, dibandingkan dengan isu pemerataan energi.

# >---

# ### **Analisis Sentimen dengan Indo-BERT**

# Di project ini, kami menggunakan model indolem/indobert-base-uncased. Model Indo-BERT adalah salah satu model BERT yang dilatih dalam bahasa Indonesia.  Model ini didapatkan dari https://indolem.github.io/IndoBERT/. Dengan detail:
# 1. Uncased, semua teks diubah menjadi huruf kecil
# 2. Ukuran data latih kurang lebih 220 juta kata
# 3. Tidak dilakukan fine tuning di dalam model ini.

# **Load model**

# In[77]:


from transformers import BertTokenizer

# Load tokenizer dari pre-trained model
bert_tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')


# In[78]:


vocabulary = bert_tokenizer.get_vocab()
print('Panjang vocabulary:', len(vocabulary))


# In[79]:


print('Kalimat:', data_label['Tweet_Text'][0])
print('BERT Tokenizer:', bert_tokenizer.tokenize(data_label['Tweet_Text'][0]))


# **Encoding (mengubah satu kalimat teks mentah menjadi format numerik yang siap dipakai model Indo-BERT)**
# 
# Prosesnya:
# 1. Mencari max_length yang paling optimal untuk tweet tokenization, agar tidak terlalu pendek dan terlalu panjang.
# 1. Mengambil satu kalimat dalam data kolom Tweet_Text[0] sampai akhir
# 2. Menambah token spesial (seperti membuat label), [CLS] di awal dan [SEP] di akhir. CLS artinya classification dan SEP adalah separator. Default dari arsitektur BERT.
# 3. Max length dari token akan dihitung di kode setelah ini.

# In[80]:


token_lens = []
for txt in data_label['Tweet_Text']:
  tokens = bert_tokenizer.encode(txt)
  token_lens.append(len(tokens))
sns.histplot(token_lens, kde=True, stat='density', linewidth=0)
plt.xlim([0, 100]);
plt.xlabel('Token count');


# Grafik ini menunjukkan bahwa sebagian besar tweet memiliki panjang di bawah 60 token. Kepadatan data sangat tinggi di rentang 20-50 token dan kemudian menurun drastis, dengan hanya sedikit sekali tweet yang memiliki panjang lebih dari 70 token. **Jadi baiknya max-length kita adalah 65**

# **Melakukan proses encoding untuk setiap tweet**

# In[81]:


def encode_teks(sentence):
  return bert_tokenizer.encode_plus(
      sentence,
      add_special_tokens=True,
      padding='max_length',
      truncation='longest_first',
      max_length=65,
      return_attention_mask=True,
      return_token_type_ids=True)


# **Helper function untuk menata ulang atau memformat data agar strukturnya tepat untuk pyTorch**

# In[82]:


#Function map
def mapping_function(input_ids, attention_masks, token_type_ids, label):
  return {
      "input_ids": input_ids,               # Sebagai token embedding
      "token_type_ids": token_type_ids,     # Sebagai segment embedding
      "attention_mask": attention_masks,     # Sebagai filter informasi mana yang kalkulasi oleh model
  }, label


# **Split dataset menjadi tiga bagian;Train, Validation dan Test**
# 1. Train (70%): Untuk melatih model
# 2. Validation set (10%) : untuk memberikan evaluasi yang tidak bias selama proses pelatihan untuk membantu kita membuat keputusan. Bisa mencegah overfitting saat kita memantau loss pada training set dan validation set. Jika training loss terus menurun sementara validation loss mulai naik, ini adalah tanda jelas bahwa model mulai "menghafal" data latih dan tidak lagi belajar pola yang umum. Berdasarkan titik ini, kita bisa menghentikan pelatihan lebih awal untuk mendapatkan model dengan kemampuan generalisasi terbaik.
# 3. Test Data (20%) : untuk memberikan penilaian akhir yang paling objektif terhadap model
# 
# * Pembagian menjadi 3 dataset dikarenakan kita menggunakan Training Set untuk mengajar model dan Validation Set untuk semua proses tuning dan pemilihan model terbaik. Dengan begitu, Test Set yang tidak pernah tersentuh sama sekali dapat memberikan skor akhir yang paling baik dan objektif tentang seberapa baik performa model pada data yang benar-benar bar

# In[84]:


from sklearn.model_selection import train_test_split

# 1. Persiapkan data
X = data_label['Tweet_Text'].values  # Feature
y_sentiment = data_label['Sentimen'].values  # Target sentimen
y_aspek = data_label['Aspek'].values  # Target aspek

# 2. Split menjadi train+val dan test (80:20)
X_trainval, X_test, y_sent_trainval, y_sent_test, y_asp_trainval, y_asp_test = train_test_split(
    X, y_sentiment, y_aspek,
    test_size=0.2,
    random_state=42,
    stratify=y_sentiment  # Stratifikasi berdasarkan sentimen
)

# 3. Split train+val menjadi train dan validation (90:10 dari total data)
X_train, X_val, y_sent_train, y_sent_val, y_asp_train, y_asp_val = train_test_split(
    X_trainval, y_sent_trainval, y_asp_trainval,
    test_size=0.125,  # 0.125 * 0.8 = 0.1 (10% dari total data)
    random_state=42,
    stratify=y_sent_trainval
)

# 4. Print distribusi data
print("Distribusi Data:")
print(f"Total Dataset: {len(X)}")
print(f"Training Set: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation Set: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test Set: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# 5. Cek distribusi label di setiap split
def print_distribution(y, title):
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n{title}:")
    for label, count in zip(unique, counts):
        print(f"{label}: {count} ({count/len(y)*100:.1f}%)")

print_distribution(y_sent_train, "Distribusi Sentimen - Training Set")
print_distribution(y_sent_val, "Distribusi Sentimen - Validation Set")
print_distribution(y_sent_test, "Distribusi Sentimen - Test Set")

# 6. Simpan hasil split (opsional)
splits = {
    'train': {
        'texts': X_train,
        'sentiment': y_sent_train,
        'aspek': y_asp_train
    },
    'val': {
        'texts': X_val,
        'sentiment': y_sent_val,
        'aspek': y_asp_val
    },
    'test': {
        'texts': X_test,
        'sentiment': y_sent_test,
        'aspek': y_asp_test
    }
}


# ### **Modelling**

# **Memuat model dasar IndoBERT**

# In[107]:


from transformers import TFBertForSequenceClassification

# Load model
bert_model = TFBertForSequenceClassification.from_pretrained(
    'indolem/indobert-base-uncased', 
    num_labels=3, #Label : Positif, Netral, Negatif
    from_pt=True  
)

print("Model berhasil dimuat dari bobot PyTorch ke TensorFlow!")


# **Konfigurasi**

# In[108]:


class Config:
    # Model settings
    MODEL_NAME = "indolem/indobert-base-uncased"  
    MAX_LENGTH = 65  
    
    # Training settings
    BATCH_SIZE = 16  
    LEARNING_RATE = 3e-5
    EPOCHS = 3
    WARMUP_RATIO = 0.1
    
    # Data settings
    TRAIN_SIZE = 0.7
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_STATE = 42
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
print(f"Using device: {config.DEVICE}")


# **Mapping label**

# In[123]:


label2id_sentiment = {'Positif': 0, 'Netral': 1, 'Negatif': 2}
label2id_aspek = {'ENERGY_ACCESS': 0, 'ENVIRONMENTAL_IMPACT': 1, 'GOVERNMENT_POLICY': 2, 'ENERGY_TECHNOLOGY': 3, 'OTHER':4} 


# In[124]:


from torch.utils.data import Dataset

class IndoBERTDataset(Dataset):
    def __init__(self, texts, sentiments, aspek, tokenizer, max_len=65):
        self.texts = texts
        self.sentiments = sentiments
        self.aspek = aspek
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )

        return {
        'input_ids': encoding['input_ids'].squeeze(),
        'attention_mask': encoding['attention_mask'].squeeze(),
        'token_type_ids': encoding['token_type_ids'].squeeze(),
        'sentiment': torch.tensor(label2id_sentiment[self.sentiments[idx]], dtype=torch.long),
        'aspek': torch.tensor(label2id_aspek[self.aspek[idx]], dtype=torch.long),
}



# In[125]:


from torch.utils.data import DataLoader

def create_dataloaders(splits, tokenizer, batch_size=16):
    train_dataset = IndoBERTDataset(splits['train']['texts'], splits['train']['sentiment'], splits['train']['aspek'], tokenizer)
    val_dataset   = IndoBERTDataset(splits['val']['texts'], splits['val']['sentiment'], splits['val']['aspek'], tokenizer)
    test_dataset  = IndoBERTDataset(splits['test']['texts'], splits['test']['sentiment'], splits['test']['aspek'], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


# In[126]:


import torch.nn as nn
from transformers import AutoModel

class MultiTaskIndoBERT(nn.Module):
    def __init__(self, model_name, num_sentiment_labels, num_aspek_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.sentiment_out = nn.Linear(self.bert.config.hidden_size, num_sentiment_labels)
        self.aspek_out = nn.Linear(self.bert.config.hidden_size, num_aspek_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        sentiment_logits = self.sentiment_out(pooled_output)
        aspek_logits = self.aspek_out(pooled_output)

        return sentiment_logits, aspek_logits


# In[127]:


def train_model(model, train_loader, val_loader, device, epochs=3, lr=2e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            sentiment = batch['sentiment'].to(device)
            aspek = batch['aspek'].to(device)

            optimizer.zero_grad()
            sent_logits, asp_logits = model(input_ids, attention_mask, token_type_ids)

            loss1 = criterion(sent_logits, sentiment)
            loss2 = criterion(asp_logits, aspek)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}")


# In[137]:


from sklearn.metrics import classification_report

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true_sent, y_pred_sent = [], []
    y_true_asp, y_pred_asp = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            sentiment = batch['sentiment'].to(device)
            aspek = batch['aspek'].to(device)

            sent_logits, asp_logits = model(input_ids, attention_mask, token_type_ids)

            sent_preds = torch.argmax(sent_logits, dim=1)
            asp_preds = torch.argmax(asp_logits, dim=1)

            y_true_sent.extend(sentiment.cpu().numpy())
            y_pred_sent.extend(sent_preds.cpu().numpy())
            y_true_asp.extend(aspek.cpu().numpy())
            y_pred_asp.extend(asp_preds.cpu().numpy())

    print("\nSentiment Classification Report:")
    print(classification_report(y_true_sent, y_pred_sent))

    print("\nAspek Classification Report:")
    print(classification_report(y_true_asp, y_pred_asp))

    return y_true_sent, y_pred_sent, y_true_asp, y_pred_asp  # ← Tambahkan ini


# In[138]:


bert_tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

train_loader, val_loader, test_loader = create_dataloaders(splits, bert_tokenizer)

model = MultiTaskIndoBERT(
    model_name=config.MODEL_NAME,
    num_sentiment_labels=len(np.unique(y_sentiment)),
    num_aspek_labels=len(np.unique(y_aspek))
)

train_model(model, train_loader, val_loader, device=config.DEVICE, epochs=3)

evaluate_model(model, test_loader, config.DEVICE)

# Menyimpan model setelah training
torch.save(model.state_dict(), "indobert_multitask.pth")
print("✅ Model berhasil disimpan sebagai indobert_multitask.pth")



# In[139]:


true_sentiments, pred_sentiments, true_aspeks, pred_aspeks = evaluate_model(model, test_loader, config.DEVICE)


# In[140]:


import pandas as pd
import numpy as np
from collections import Counter

# Load dataset
df = pd.read_csv("labeling_claude.csv", sep=';', encoding='utf-8')  
print("🚀 INDOBERT SENTIMENT ANALYSIS FINE-TUNING")
print("=" * 60)

# Informasi dataset
print("📊 Loading labeled dataset...")
print(f"✅ Dataset loaded: {df.shape}")
print(f"✅ Columns: {list(df.columns)}\n")

# Panjang teks
text_lengths = df['Tweet_Text'].apply(len)
word_counts = df['Tweet_Text'].apply(lambda x: len(str(x).split()))

print("📈 DATASET ANALYSIS:")
print("-" * 50)
print(f"Text length - Mean: {text_lengths.mean():.1f}, Max: {text_lengths.max()}")
print(f"Word count - Mean: {word_counts.mean():.1f}, Max: {word_counts.max()}")
long_texts = (text_lengths > 512).sum()
print(f"Texts > 512 chars: {long_texts} ({100 * long_texts / len(df):.1f}%)\n")

# Distribusi Sentimen
print("🎯 SENTIMENT DISTRIBUTION:")
sentiment_counts = Counter(df['Sentimen'])
for label, count in sentiment_counts.items():
    print(f"  {label.capitalize()}: {count} ({100 * count / len(df):.1f}%)")

# Distribusi Aspek
print("\n🎯 ASPEK DISTRIBUTION:")
aspek_counts = Counter(df['Aspek'])
for label, count in aspek_counts.items():
    print(f"  {label.upper()}: {count} ({100 * count / len(df):.1f}%)")


# In[142]:


import pandas as pd

df_llm = pd.read_csv('labeling_claude.csv', sep=';')

# Pastikan mapping label Claude sesuai label IndoBERT
# Misal: 'Netral' → 0, 'Negatif' → 1, 'Positif' → 2
sentiment_map = {'Netral': 0, 'Negatif': 1, 'Positif': 2}
aspek_map = {
    'ENVIRONMENTAL_IMPACT': 0,
    'ENERGY_TECHNOLOGY': 1,
    'OTHER': 2,
    'GOVERNMENT_POLICY': 3,
    'ENERGY_ACCESS': 4
}

llm_sentiments = df_llm['Sentimen'].map(sentiment_map).tolist()
llm_aspeks = df_llm['Aspek'].map(aspek_map).tolist()


# In[145]:


sentiment_map = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
aspek_map = {
    'ENVIRONMENTAL_IMPACT': 0,
    'ENERGY_TECHNOLOGY': 1,
    'OTHER': 2,
    'GOVERNMENT_POLICY': 3,
    'ENERGY_ACCESS': 4
}

llm_sentiments = llm_test['Sentimen'].map(sentiment_map).tolist()
llm_aspeks = llm_test['Aspek'].map(aspek_map).tolist()


# In[152]:


true_sentiments, pred_sentiments, true_aspeks, pred_aspeks = evaluate_model(
    model,
    test_loader,
    config.DEVICE
)

