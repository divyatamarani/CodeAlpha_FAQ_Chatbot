import streamlit as st
import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# ---------- Page Config ----------
st.set_page_config(page_title="AI FAQ Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 AI FAQ Chatbot")
st.write("Ask a question related to Artificial Intelligence, Python, Git, NLP, etc.")

# ---------- Load FAQ ----------
faq = pd.read_csv("faq.csv")

# ---------- Text Preprocessing ----------
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

faq["Processed"] = faq["Question"].apply(preprocess)

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq["Processed"])

# ---------- Chat History ----------
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask your question:")

if st.button("Get Answer"):

    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        processed_query = preprocess(query)
        query_vector = vectorizer.transform([processed_query])

        similarity = cosine_similarity(query_vector, faq_vectors)

        best_match = similarity.argmax()
        score = similarity[0][best_match]

        if score > 0.20:
            answer = faq.iloc[best_match]["Answer"]
        else:
            answer = "Sorry, I couldn't find a suitable answer."

        st.session_state.history.append((query, answer))

        st.success(answer)

# ---------- Chat History ----------
if st.session_state.history:

    st.markdown("---")
    st.subheader("💬 Chat History")

    for q, a in reversed(st.session_state.history):
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")
        st.write("")