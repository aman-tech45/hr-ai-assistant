# 💼 AI-Powered HR Assistant (RAG-Based Chatbot)

An intelligent HR chatbot that answers employee queries using company policy documents. Built using Retrieval-Augmented Generation (RAG) with semantic search.

---

## 🚀 Live Demo

🔗 *(Add your Streamlit link here after deployment)*

---

## 📌 Features

* 📄 Understands HR policy documents (PDF)
* 🔍 Semantic search using embeddings (not keyword-based)
* ⚡ Fast retrieval with FAISS vector database
* 🧠 Intelligent filtering using section-based classification
* 💬 Interactive chatbot UI built with Streamlit
* 🆓 Uses local embeddings (no API cost)

---

## 🧠 How It Works

1. PDF is loaded and cleaned
2. Text is split into chunks
3. Each chunk is converted into embeddings
4. Stored in FAISS vector database
5. User query → semantic search → relevant chunks retrieved
6. System generates structured response

---

## 🏗️ Tech Stack

* Python
* LangChain
* FAISS (Vector DB)
* HuggingFace Embeddings
* Streamlit (UI)

---

## 📂 Project Structure

```
hr-ai-assistant/
│
├── ui.py               # Streamlit frontend
├── app.py              # Backend RAG pipeline
├── hr_policy.pdf       # HR policy document
├── requirements.txt    # Dependencies
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/aman-tech45/hr-ai-assistant.git
cd hr-ai-assistant

pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
streamlit run ui.py
```

---

## 📊 Example Queries

* What is leave policy?
* What are attendance rules?
* How many leaves are allowed?
* What is probation period?

---

## 🧠 Key Concepts Used

* Retrieval-Augmented Generation (RAG)
* Embeddings for semantic understanding
* Vector similarity search
* Chunking strategies for document processing

---

## 🚀 Future Improvements

* Add LLM (ChatGPT-style answers)
* Add PDF upload feature
* Add chat history
* Improve UI/UX design
* Add source citations (page-level)

---

## 👨‍💻 Author

**Aman Jha**
📍 Mumbai, India
🔗 https://github.com/aman-tech45

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
