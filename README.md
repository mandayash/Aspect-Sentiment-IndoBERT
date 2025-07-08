# Analisis Aspek dan Sentimen Diskusi Green Economy dan EBT di Media Sosial X

## 📋 Deskripsi Project

Penelitian ini menganalisis sentimen dan aspek diskusi Green Economy dan Energi Baru Terbarukan (EBT) di media sosial X (Twitter) menggunakan pendekatan **Multi-Task Learning dengan IndoBERT**. Project ini menggabungkan **Topic Modeling LDA** untuk mengidentifikasi tema diskusi dan **Multi-Task IndoBERT** untuk klasifikasi sentimen dan aspek secara simultan.

### 🎯 Tujuan Penelitian
- Menganalisis pola sentimen publik terhadap diskusi EBT di media sosial X
- Mengidentifikasi aspek dominan dalam diskusi Green Economy dan EBT di Indonesia  
- Mengevaluasi efektivitas IndoBERT multi-task learning untuk analisis aspek dan sentimen

## 🔧 Teknologi & Framework

- **Python 3.8+**
- **PyTorch** - Deep Learning Framework
- **HuggingFace Transformers** - Pre-trained Models
- **Scikit-learn** - Machine Learning Tools
- **Gensim** - Topic Modeling LDA
- **Pandas & NumPy** - Data Processing
- **Matplotlib & Seaborn** - Visualisasi

## 📊 Dataset

### Data Collection
- **Platform**: Twitter/X menggunakan Twitter API v2
- **Periode**: 28 Desember 2024 - 2 Januari 2025
- **Total Tweets**: 949 unique tweets (dari 1000+ raw data)
- **Keywords**: 20 terms terkait EBT (Indonesia + English)

### Core Themes (5 Aspek)
1. **ENERGY_TECHNOLOGY** - Teknologi energi terbarukan dan transisi energi
2. **ENVIRONMENTAL_IMPACT** - Dampak lingkungan, deforestasi, kelapa sawit
3. **GOVERNMENT_POLICY** - Kebijakan pemerintah dan tokoh politik
4. **ENERGY_ACCESS** - Akses energi untuk masyarakat
5. **OTHER** - Topik umum dan noise

### Label Distribution
- **Sentimen**: Positif (31.2%), Negatif (43.8%), Netral (25.0%)
- **Aspek**: Environmental Impact (45.0%), Energy Technology (21.0%), Other (14.2%), Government Policy (12.4%), Energy Access (7.4%)

## 📈 Hasil Utama

### Model Performance
- **Sentiment Classification**: 89% accuracy
- **Aspect Classification**: 92.6% overall agreement
- **LLM-Human Agreement**: 88.9% (sentiment), 88.4% (aspect)

### Topic Modeling Results
- **Optimal Topics**: 11 topics (LDA)
- **Coherence Score**: 0.6576 (excellent)
- **Core Themes**: Successfully mapped to 5 main aspects

### Key Insights
- **Environmental Impact** mendominasi diskusi (45.0%)
- **Sentimen negatif** lebih prevalent (43.8%) 
- **Energy Access** underrepresented (7.4%) - communication gap
- **Perfect classification** untuk Energy Access meskipun data terbatas

## 🔍 Penggunaan

### Basic Usage
```python
from src.models.multitask_bert import MultiTaskIndoBERT
from transformers import AutoTokenizer

# Load trained model
model = MultiTaskIndoBERT.load_from_checkpoint('models/indobert_multitask.pth')
tokenizer = AutoTokenizer.from_pretrained('indolem/indobert-base-uncased')

# Predict sentiment and aspect
text = "Panel surya sangat membantu untuk energi terbarukan di Indonesia"
sentiment, aspect = model.predict(text, tokenizer)
print(f"Sentiment: {sentiment}, Aspect: {aspect}")
```

### Dataset Requirements
Untuk menjalankan full pipeline, Anda membutuhkan:

1. **Raw tweets** (hasil scraping) atau gunakan `hasil-scraping.csv`
2. **Preprocessed data** atau jalankan preprocessing pipeline
3. **Labeled data** untuk training atau gunakan `labelling_claude.csv`

## 📄 Dataset & Model

### Available Files
- `hasil-scraping.csv` - Raw scraped tweets
- `ready_labelling.csv` - Preprocessed tweets  
- `labelling_claude.csv` - Labeled ground truth
- `indobert_multitask.pth` - Trained model

### Data Usage
Dataset ini dapat digunakan untuk:
- Sentiment analysis research
- Aspect-based sentiment analysis
- Indonesian NLP tasks
- Policy communication analysis

## 🙏 Credits & References

### Team Members
1. **Amanda Putri Aprilliani** (105222001)
2. **Raihan Akira Rahmaputra** (105222040) 
3. **Gema Fitri Ramadani** (105222009)
4. **Anom Wajawening** (105222029)

### Key References
- IndoBERT: [IndoLEM and IndoBERT Paper](https://aclanthology.org/2020.coling-main.66/)
- Multi-Task Learning: [HydraNets with PyTorch](https://pyimagesearch.com/2022/08/17/multi-task-learning-and-hydranets-with-pytorch/)
- Topic Modeling: Latent Dirichlet Allocation implementation

## 📞 Contact

Untuk pertanyaan atau kolaborasi:
- LinkedIn: www.linkedin.com/in/amandaputriapr

## 📝 License

Silakan gunakan untuk penelitian dan pengembangan lebih lanjut, sertakan nama pengembang.

⭐ **Jika project ini helpful, jangan lupa berikan star!** ⭐
