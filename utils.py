import re
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load English stop words
stop_words = set(stopwords.words('english'))

def parse_chat_log(file_path):
    """Parses a single chat log file and separates user and AI messages."""
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
    """Returns a dictionary of message counts."""
    return {
        "total_messages": len(user_msgs) + len(ai_msgs),
        "user_messages": len(user_msgs),
        "ai_messages": len(ai_msgs)
    }

def extract_keywords(messages, top_n=5, use_tfidf=False):
    """
    Extracts top keywords from a list of messages using word frequency or TF-IDF.
    
    Parameters:
    - messages: list of strings (user or AI messages)
    - top_n: number of keywords to extract
    - use_tfidf: whether to use TF-IDF (True) or word frequency (False)
    
    Returns:
    - List of top keywords (strings)
    """
    text = " ".join(messages).lower()

    if use_tfidf:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
        tfidf = vectorizer.fit_transform([text])
        return vectorizer.get_feature_names_out().tolist()

    # Word frequency approach
    words = re.findall(r'\b\w+\b', text)
    filtered = [w for w in words if w not in stop_words]
    counter = Counter(filtered)
    return [word for word, _ in counter.most_common(top_n)]
