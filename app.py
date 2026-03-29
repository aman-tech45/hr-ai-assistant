import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------
# 1. CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -------------------------------
# 2. LOAD DOCUMENT
# -------------------------------
loader = PyPDFLoader("hr_policy.pdf")
documents = loader.load()

for doc in documents:
    doc.page_content = clean_text(doc.page_content)

print(f"✅ Pages loaded: {len(documents)}")

# -------------------------------
# 3. SECTION DETECTION
# -------------------------------
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
    elif "recruitment" in t:
        return "recruitment"
    elif "probation" in t:
        return "probation"
    elif "salary" in t or "wage" in t:
        return "salary"
    else:
        return "general"

# -------------------------------
# 4. CHUNKING
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

# Attach metadata
for chunk in chunks:
    chunk.metadata["section"] = get_section(chunk.page_content)

print(f"✅ Chunks created: {len(chunks)}")

# -------------------------------
# 5. EMBEDDINGS
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -------------------------------
# 6. VECTOR STORE + RETRIEVER
# -------------------------------
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10}
)

print("✅ Vector store ready!")

# -------------------------------
# 7. QUERY CLASSIFIER
# -------------------------------
def classify_query(query):
    q = query.lower()

    if "leave" in q:
        return "leave"
    elif "attendance" in q:
        return "attendance"
    elif "salary" in q or "wage" in q:
        return "salary"
    elif "travel" in q:
        return "travel"
    elif "performance" in q:
        return "performance"
    elif "probation" in q:
        return "probation"
    elif "recruitment" in q or "hiring" in q:
        return "recruitment"
    else:
        return "general"

# -------------------------------
# 8. CHATBOT FUNCTION
# -------------------------------
def ask_hr_bot(query):

    # Filter non-HR queries
    if any(word in query.lower() for word in ["pip", "install", "python", "code"]):
        return "⚠️ Please ask HR-related questions only."

    intent = classify_query(query)

    docs = retriever.invoke(query)

    # Filter by section
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

    # fallback
    if not insights:
        insights = [doc.page_content[:200] for doc in filtered_docs]

    # remove duplicates
    insights = list(dict.fromkeys(insights))

    # FIXED f-string issue
    bullet_points = "\n- ".join(insights[:5])

    return f"""
🤖 HR Assistant

📌 Question:
{query}

🧠 Topic: {intent.upper()}

✅ Key Insights:
- {bullet_points}

📄 Source:
Company HR Policy Document
"""

# -------------------------------
# 9. INTERACTIVE LOOP
# -------------------------------
print("\n🚀 HR Assistant Ready!\n")

while True:
    user_query = input("👉 Ask HR Question: ").strip()

    if not user_query:
        print("⚠️ Please enter a valid question.")
        continue

    if user_query.lower() == "exit":
        print("👋 Exiting...")
        break

    response = ask_hr_bot(user_query)

    print("\n" + "="*60)
    print(response)
    print("="*60)