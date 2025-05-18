# AI Chat Log Summarizer

A Python tool that reads `.txt` chat logs between a user and an AI, parses the conversation, and produces a summary with message counts and key topics.

---

## Features

- Parses chat logs formatted with `User:` and `AI:` messages.
- Counts total messages, and messages by each speaker.
- Extracts top 5 keywords excluding common stopwords.
- Optional TF-IDF-based keyword extraction for improved keyword relevance.
- Processes multiple chat logs from a folder.

---

## How I Developed This Project

1. First, I created an initial chat log file (`chat1.txt`) containing a sample conversation.
2. Then, I added another chat log file (`chat2.txt`) to implement the optional feature of processing multiple chat logs.
3. I developed `utils.py` to:
   - Parse chat logs into separate user and AI message lists.
   - Calculate message statistics.
   - Extract keywords using simple frequency counts and optional TF-IDF (using `scikit-learn`).
4. Next, I created `summarizer.py` to:
   - Iterate over all chat log files in the `chat_logs/` folder.
   - Summarize each conversation by printing message counts and top keywords.
5. I prepared `requirements.txt` listing the necessary Python packages (`nltk`, `scikit-learn`, `numpy`).
6. Finally, I wrote this `README.md` to document the project, usage instructions, and development process.

---

## Usage

1. I placed all chat `.txt` files inside the `chat_logs/` folder within the project directory.

2. I installed the required dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

---

## Sample Output

Here is an example of the summarizer output:

![Summarizer Output](screenshot/Screenshot_output.png)
