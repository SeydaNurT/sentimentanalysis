from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
import string


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

app = Flask(__name__)

model = joblib.load("sentiment_nb_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

stop_words = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can't","cannot","could","couldn't","did",
    "didn't","do","does","doesn't","doing","don't","down","during","each",
    "few","for","from","further","had","hadn't","has","hasn't","have",
    "haven't","having","he","he'd","he'll","he's","her","here","here's",
    "hers","herself","him","himself","his","how","how's","i","i'd","i'll",
    "i'm","i've","if","in","into","is","isn't","it","it's","its","itself",
    "let's","me","more","most","mustn't","my","myself","no","nor","not",
    "of","off","on","once","only","or","other","ought","our","ours",
    "ourselves","out","over","own","same","shan't","she","she'd","she'll",
    "she's","should","shouldn't","so","some","such","than","that","that's",
    "the","their","theirs","them","themselves","then","there","there's",
    "these","they","they'd","they'll","they're","they've","this","those",
    "through","to","too","under","until","up","very","was","wasn't","we",
    "we'd","we'll","we're","we've","were","weren't","what","what's",
    "when","when's","where","where's","which","while","who","who's",
    "whom","why","why's","with","won't","would","wouldn't","you","you'd",
    "you'll","you're","you've","your","yours","yourself","yourselves"
}

punctuations = string.punctuation

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words and w not in punctuations]
    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        user_text = request.form.get("review")
        cleaned = clean_text(user_text)
        vector = tfidf.transform([cleaned])
        result = model.predict(vector)[0]

        prediction = "Positive 🙂" if result == 1 else "Negative 🙁"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
