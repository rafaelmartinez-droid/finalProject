import streamlit as st
import chromadb
from groq import Groq
from pypdf import PdfReader
from datetime import datetime

now = datetime.now()

now = str(now).replace(":", "_")
now = now.replace(" ", "_")

API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=API_KEY)
MODEL = "llama-3.1-8b-instant"

with st.sidebar:
    st.title("Settings")
    print_distances = st.checkbox("Display distances")
    print_file_type = st.checkbox("Display file type")
    print_chunk_amount = st.checkbox("Display chunk amount")
    chunk_size = st.slider("Chunk size", min_value=100, max_value=1000, value=300, step=10)
    overlap = st.slider("Overlap", min_value=0, max_value=chunk_size-10, value=chunk_size-100, step=10)
    distance_limiter = st.slider("Accept answers below this distance", min_value=0.0, max_value=3.0, value=1.25, step=0.01)
    amount_of_results = st.slider("Number of results", min_value=1, max_value=25, value=10, step=1)
    st.write("current chunk size:", chunk_size)
    st.write("current overlap:", overlap)
    st.write("current distance limiter:", distance_limiter)
    st.write("current amount of results:", amount_of_results)

#file = st.file_uploader("Upload a .txt file", type="txt")#change
st.header("LLM and RAG document interpreter")
file = st.file_uploader("1. Upload a .pdf file or a .txt file", type=["pdf", "txt"])#change

if file and st.button("Process File"):
    st.write("Processing file")
    if print_file_type:
        st.write("File type is: ", file.type)
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
    chunks = []
    # chunk_size = 300
    # overlap = 100
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i: i + chunk_size])
    if print_chunk_amount:
        st.write("There are: ", len(chunks), " chunks")
    chroma_client = chromadb.Client()
    st.session_state.chroma_client = chroma_client
    #collection = chroma_client.create_collection("documents" + file.name)
#try:
    collection = chroma_client.create_collection("document_" + now + "_" + file.name)
#except Exception:
 #   collection = chroma_client.get_collection("testing" + now)
    st.session_state.collection = collection
    tags = [file.name + str(i) for i in range(len(chunks))] #better citations
    collection.add(documents=chunks, ids=tags)

    st.session_state.collection = collection
    st.write("Chunks added to knowledge base!")

question = st.text_input("2. Ask a question about the file")



if st.button("3. Search"):
    if "collection" not in st.session_state:
        st.write("No documents provided")

    else:
        st.write("Thinking!")
        collection = st.session_state.collection

        result = collection.query(query_texts=[question], n_results=amount_of_results)

        if "context" not in st.session_state:
            st.session_state.context = []
        for i in result["documents"][0]:
            rd = result["documents"][0]
            if result["distances"][0][rd.index(i)] < distance_limiter:
                st.session_state.context.append(i)
                # st.write(i)
            else:
                continue
        if st.session_state.context == []:
            st.write("This question was not answered in the document provided")
        st.session_state.question = question
        if print_distances:
            st.write(result["distances"])
        # st.write(st.session_state.context)
        # for ans in st.session_state.context:
        #     st.write(ans)
        st.write("Done!")

if st.button("4. LLM answer"):
    if "question" in st.session_state:
        st.write("Contacting LLM...")

        context = "\n".join(st.session_state.context)
        question = st.session_state.question

        messages = [
            {"role": "system", "content": "Answer the user's question using only the provided document context. If the context contains enough information to answer, give the answer."},
            {"role": "user", "content": f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}"}
        ]

        response = client.chat.completions.create(model=MODEL, messages=messages)
        st.write("LLM Answer:", response.choices[0].message.content)
    else:
        st.write("No question provided")
if st.button("Delete collection", icon="🗑️"):
    try:
        collections = st.session_state.chroma_client.list_collections()
        for collection in collections:
            if "document_" in collection.name:
                st.session_state.chroma_client.delete_collection(name=collection.name)
                st.write(f"Deleted collection: {collection.name}")
        st.session_state.context = []
        del st.session_state['collection']
        del st.session_state['question']
    except Exception:
        st.write("Nothing found to delete")
        st.session_state.context = []
