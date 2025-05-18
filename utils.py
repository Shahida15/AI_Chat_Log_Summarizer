import re
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = set(stopwords.words('english'))

def parse_chat_log(file_path):
    user_msgs, ai_msgs = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("User:"):
                user_msgs.append(line[5:].strip())
            elif line.startswith("AI:"):
                ai_msgs.append(line[3:].strip())
    return user_msgs, ai_msgs

def get_message_stats(user_msgs, ai_msgs):
    return {
        "total_messages": len(user_msgs) + len(ai_msgs),
        "user_messages": len(user_msgs),
        "ai_messages": len(ai_msgs)
    }

def extract_keywords(messages, top_n=5, use_tfidf=False):
    text = " ".join(messages).lower()
    words = re.findall(r'\b\w+\b', text)
    filtered = [w for w in words if w not in stop_words]

    if use_tfidf:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
        tfidf = vectorizer.fit_transform([text])
        return vectorizer.get_feature_names_out().tolist()

    counter = Counter(filtered)
    return [word for word, _ in counter.most_common(top_n)]