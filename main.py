import streamlit as st
import chromadb
from groq import Groq
from pypdf import PdfReader
from datetime import datetime

API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=API_KEY)
MODEL = "llama-3.1-8b-instant"

#file = st.file_uploader("Upload a .txt file", type="txt")#change
file = st.file_uploader("Upload a .pdf file", type=["pdf", "txt"])#change

if file and st.button("Process File"):
    st.write("File processed")
    st.write(file.type)
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
    chunks = []
    chunk_size = 300
    overlap = 100
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i: i + chunk_size])
    st.write(len(chunks))
    now = datetime.now()
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(str(now) + "documents")
    tags = [file.name + str(i) for i in range(len(chunks))] #better citations
    collection.add(documents=chunks, ids=tags)

    st.session_state.collection = collection
    st.write("Chunks added to knowledge base!")

question = st.text_input("Ask a question about the file")

if st.button("Search"):
    st.write("thinking!")
    collection = st.session_state.collection
    result = collection.query(query_texts=[question], n_results=10)
    if "context" not in st.session_state:
        st.session_state.context = []
    for i in result["documents"][0]:
        rd = result["documents"][0]
        if result["distances"][0][rd.index(i)] < 1.25:
            st.session_state.context.append(i)
            st.write(i)
        else:
            continue
    st.session_state.question = question
    st.write(result["distances"])
    st.write(st.session_state.context)
    for ans in st.session_state.context:
        st.write(ans)

if st.button("LLM answer"):
    st.write("contacting LLM...")

    context = "\n".join(st.session_state.context)
    question = st.session_state.question

    messages = [
        {"role": "system", "content": "Answer the user's question using only the provided document context. If the context contains enough information to answer, give the answer."},
        {"role": "user", "content": f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}"}
    ]

    response = client.chat.completions.create(model=MODEL, messages=messages)
    st.write("LLM Answer:", response.choices[0].message.content)