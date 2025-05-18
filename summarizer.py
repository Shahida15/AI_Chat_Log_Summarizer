import os
from utils import parse_chat_log, get_message_stats, extract_keywords

def summarize_chat(file_path, use_tfidf=False):
    """Summarizes a single chat log with message stats and keyword extraction."""
    user_msgs, ai_msgs = parse_chat_log(file_path)
    stats = get_message_stats(user_msgs, ai_msgs)
    keywords = extract_keywords(user_msgs + ai_msgs, top_n=5, use_tfidf=use_tfidf)

    print("\nSummary:")
    print(f"- Total messages: {stats['total_messages']}")
    print(f"- User messages: {stats['user_messages']}, AI messages: {stats['ai_messages']}")
    print(f"- Most common keywords: {', '.join(keywords)}")

def process_all_chats(folder_path, use_tfidf=False):
    """Processes all .txt chat logs in the specified folder."""
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            file_path = os.path.join(folder_path, fname)
            print(f"\n== Chat File: {fname} ==")
            summarize_chat(file_path, use_tfidf=use_tfidf)

if __name__ == "__main__":
    chat_dir = "chat_logs"
    # Toggle use_tfidf=True to enable TF-IDF keyword extraction
    process_all_chats(chat_dir, use_tfidf=True)
