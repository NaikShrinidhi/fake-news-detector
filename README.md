📰 Fake News Detector

A machine learning system that classifies news articles as Real or Fake using TF-IDF vectorization and a Passive Aggressive Classifier — deployed as an interactive Streamlit web app.

🔗 Live Demo: [Add your Streamlit Cloud link here after deployment]

Problem Statement

Misinformation spreads rapidly through digital news and social media, making it increasingly difficult for readers to distinguish credible reporting from fabricated content. This project builds a lightweight, fast text-classification model that flags likely fake news articles based purely on their textual content — no metadata, no source reputation lookup, just the language of the article itself.

Dataset

Fake and Real News Dataset by Clément Bisaillon (Kaggle)

Total articles: 44,898 (23,481 fake / 21,417 real)
Class balance: ~52% fake, ~48% real
No missing values in title, text, subject, or date columns
Approach
Data preparation — combined and shuffled labeled Fake/Real CSVs into a single dataframe
EDA — verified class balance, checked for missing values, examined article length distribution (median ~362 words per article)
Train/test split — 80/20 (35,918 train / 8,980 test), stratified on label to preserve class ratio
Feature extraction — TF-IDF vectorization (English stop words removed, max_df=0.7 to filter overly common terms), producing 111,073 features
Model — Passive Aggressive Classifier, an online linear classifier well-suited to high-dimensional sparse text features; updates only on misclassified or low-margin examples
Evaluation — accuracy, F1 score, and full classification report on held-out test data
Persistence — trained model and fitted vectorizer saved separately via joblib, loaded together at inference time (the vectorizer's fitted vocabulary must match the model's expected feature space)
Deployment — wrapped in a Streamlit UI, deployed on Streamlit Community Cloud
Results
Metric	Score
Accuracy	99.35%
F1 Score	0.9932
Test set size	8,980 articles
TF-IDF features	111,073

Classification Report:

Class	Precision	Recall	F1-score	Support
Fake	1.00	0.99	0.99	4,696
Real	0.99	0.99	0.99	4,284
Tech Stack
Language: Python
ML/Data: scikit-learn, pandas, NumPy
Visualization: Matplotlib, Seaborn
Model persistence: joblib
Web app: Streamlit
Deployment: Streamlit Community Cloud
How to Run Locally
bash
git clone https://github.com/NaikShrinidhi/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
streamlit run app.py

Note: data/Fake.csv and data/True.csv are not included in this repo due to size — download them from the Kaggle dataset link above and place them in a data/ folder if you want to re-run the training notebook. The trained model and vectorizer (model/) are already included, so the app runs out of the box without needing the raw dataset.

Limitations & What I'd Improve Next
Confidence score is derived from the classifier's decision boundary distance (via decision_function), not a calibrated probability — PassiveAggressiveClassifier doesn't natively support predict_proba. A next step would be wrapping it with CalibratedClassifierCV for genuine probability outputs.
Trained on a specific dataset (primarily US political news, ~2016–2017 era) — may not generalize equally well to other topics, regions, or more recent writing styles.
Could compare against other classifiers (Logistic Regression, Naive Bayes, SVM) to benchmark performance.
Could incorporate n-grams or additional features (e.g. article length, punctuation patterns) beyond unigram TF-IDF.
Author

Shrinidhi Naik — final-year IT graduate
