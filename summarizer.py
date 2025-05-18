import os
from utils import parse_chat_log, get_message_stats, extract_keywords

def summarize_chat(file_path, summary_num):
    user_msgs, ai_msgs = parse_chat_log(file_path)
    stats = get_message_stats(user_msgs, ai_msgs)
    keywords = extract_keywords(user_msgs + ai_msgs)
    print(f"\nSummary{summary_num}:")
    print(f"- Total messages: {stats['total_messages']}")
    print(f"- User messages: {stats['user_messages']}, AI messages: {stats['ai_messages']}")
    print(f"- Most common keywords: {', '.join(keywords)}")

if __name__ == "__main__":
    chat_dir = "chat_logs"
    summary_counter = 1
    for fname in os.listdir(chat_dir):
        if fname.endswith(".txt"):
            print(f"\n== Chat File: {fname} ==")
            summarize_chat(os.path.join(chat_dir, fname), summary_counter)
            summary_counter += 1

