import streamlit as st
import joblib

# --- Load model and vectorizer ONCE when the app starts ---
# Cached so Streamlit doesn't reload these from disk on every button click
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/fake_news_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# --- Page setup ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write("Paste a news article or headline below to check if it's likely Real or Fake.")

# --- Input ---
user_input = st.text_area("Article text:", height=200, placeholder="Paste article text here...")

# --- Predict button ---
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please paste some text before predicting.")
    else:
        # Transform user text using the SAME fitted vectorizer from training
        # (transform only — never fit_transform here, no new learning happens)
        input_tfidf = vectorizer.transform([user_input])

        # Predict label: 0 = Fake, 1 = Real
        prediction = model.predict(input_tfidf)[0]

        # PassiveAggressiveClassifier has no predict_proba by default,
        # so we use decision_function as a confidence proxy —
        # it's the signed distance from the decision boundary;
        # larger absolute value = more confident
        decision_score = model.decision_function(input_tfidf)[0]
        confidence = min(abs(decision_score) / 5, 1.0) * 100  # rough scaling for display

        # --- Display result ---
        if prediction == 1:
            st.success(f"✅ This looks like REAL news")
        else:
            st.error(f"🚫 This looks like FAKE news")

        st.write(f"Confidence (based on decision distance): **{confidence:.1f}%**")
        st.caption(f"Raw decision score: {decision_score:.3f}")