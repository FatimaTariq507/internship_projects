import re
from nltk.corpus import stopwords


def clean_text(text):
    try:
        print("🔹 Cleaning:", text[:60])  # Show first part of review
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = text.split()
        words = [word for word in words if word not in stopwords.words('english')]
        clean = " ".join(words)
        print("✅ Done:", clean[:60])
        return clean
    except Exception as e:
        print("❌ Error while cleaning:", e)
        return ""

print("✅ Preprocessing check done!")
