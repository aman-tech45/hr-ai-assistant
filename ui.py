import streamlit as st
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI HR Assistant", page_icon="💼")

st.title("💼 AI-Powered HR Assistant")
st.write("Ask any HR-related question from company policy")

# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -------------------------------
# LOAD + PROCESS DOCUMENT (CACHE)
# -------------------------------
@st.cache_resource
def load_vectorstore():

    loader = PyPDFLoader("hr_policy.pdf")
    documents = loader.load()

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    # Section tagging
    def get_section(text):
        t = text.lower()
        if "leave policy" in t:
            return "leave"
        elif "attendance" in t:
            return "attendance"
        elif "travel policy" in t:
            return "travel"
        elif "performance" in t:
            return "performance"
        else:
            return "general"

    for chunk in chunks:
        chunk.metadata["section"] = get_section(chunk.page_content)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )

    return retriever

retriever = load_vectorstore()

# -------------------------------
# QUERY CLASSIFIER
# -------------------------------
def classify_query(query):
    q = query.lower()
    if "leave" in q:
        return "leave"
    elif "attendance" in q:
        return "attendance"
    elif "travel" in q:
        return "travel"
    elif "performance" in q:
        return "performance"
    else:
        return "general"

# -------------------------------
# CHAT FUNCTION
# -------------------------------
def ask_hr_bot(query):

    if any(word in query.lower() for word in ["pip", "install", "python"]):
        return "⚠️ Please ask HR-related questions only."

    intent = classify_query(query)
    docs = retriever.invoke(query)

    filtered_docs = [
        doc for doc in docs
        if doc.metadata.get("section") == intent or intent == "general"
    ]

    if not filtered_docs:
        return "❌ No relevant HR policy found."

    insights = []

    for doc in filtered_docs:
        sentences = doc.page_content.split(". ")
        for sentence in sentences:
            if any(word in sentence.lower() for word in query.lower().split()):
                if len(sentence) > 40:
                    insights.append(sentence.strip())

    if not insights:
        insights = [doc.page_content[:200] for doc in filtered_docs]

    insights = list(dict.fromkeys(insights))
    bullet_points = "\n- ".join(insights[:5])

    return f"""
🤖 HR Assistant

📌 Question:
{query}

🧠 Topic: {intent.upper()}

✅ Key Insights:
- {bullet_points}
"""

# -------------------------------
# UI INPUT
# -------------------------------
query = st.text_input("💬 Ask your HR question:")

if query:
    with st.spinner("Thinking..."):
        answer = ask_hr_bot(query)

    st.success("Answer Generated ✅")
    st.text(answer)