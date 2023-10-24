import os
import uuid
import tempfile
import streamlit as st
import openai
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import uuid
from langchain.schema.document import Document
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.document_loaders import PyPDFLoader
# Set OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set in environment variables!")
    raise SystemExit
openai.api_key = OPENAI_API_KEY


def process_pdf(uploaded_file):
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
            loaders = [PyPDFLoader(tmp_path)]
            docs = []
            for l in loaders:
                docs.extend(l.load())
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
            docs = text_splitter.split_documents(docs)
    return docs


def smaller_chunks_strategy(docs):
    with st.spinner('Processing with smaller_chunks_strategy'):
        vectorstore = Chroma(
            collection_name="full_documents",
            embedding_function=OpenAIEmbeddings()
        )
        store = InMemoryStore()
        id_key = "doc_id"
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )
        doc_ids = [str(uuid.uuid4()) for _ in docs]
        child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        sub_docs = []
        for i, doc in enumerate(docs):
            _id = doc_ids[i]
            _sub_docs = child_text_splitter.split_documents([doc])
            for _doc in _sub_docs:
                _doc.metadata[id_key] = _id
            sub_docs.extend(_sub_docs)

        retriever.vectorstore.add_documents(sub_docs)
        retriever.docstore.mset(list(zip(doc_ids, docs)))
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever, memory=memory)
    prompt = st.text_input("Enter Your Question:", placeholder="Ask something", key="1")
    if prompt:
        st.info(prompt, icon="🧐")
        result = qa({"question": prompt})
        st.success(result['answer'], icon="🤖")


def summary_strategy(docs):
    with st.spinner('Processing with summary_strategy'):
        chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
            | ChatOpenAI(max_retries=0)
            | StrOutputParser()
        )
        summaries = chain.batch(docs, {"max_concurrency": 5})
        vectorstore = Chroma(
            collection_name="summaries",
            embedding_function= OpenAIEmbeddings()
        )
        store = InMemoryStore()
        id_key = "doc_id"
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )
        doc_ids = [str(uuid.uuid4()) for _ in docs]
        summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(summaries)]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, docs)))
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever, memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True))
    prompt = st.text_input("Enter Your Question:", placeholder="Ask something", key="2")
    if prompt:
        st.info(prompt, icon="🧐")
        result = qa({"question": prompt})
        st.success(result['answer'], icon="🤖")


def hypothetical_questions_strategy(docs):
    with st.spinner('Processing with hypothetical_questions_strategy'):
        functions = [
            {
                "name": "hypothetical_questions",
                "description": "Generate hypothetical questions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                        },
                    },
                    "required": ["questions"]
                }
            }
        ]
        chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("Generate a list of 3 hypothetical questions that the below document could be used to answer:\n\n{doc}")
            | ChatOpenAI(max_retries=0, model="gpt-4").bind(functions=functions, function_call={"name": "hypothetical_questions"})
            | JsonKeyOutputFunctionsParser(key_name="questions")
        )
        hypothetical_questions = chain.batch(docs, {"max_concurrency": 5})
        vectorstore = Chroma(
            collection_name="hypo-questions",
            embedding_function=OpenAIEmbeddings()
        )
        store = InMemoryStore()
        id_key = "doc_id"
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )
        doc_ids = [str(uuid.uuid4()) for _ in docs]
        question_docs = []
        for i, question_list in enumerate(hypothetical_questions):
            question_docs.extend([Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list])
        retriever.vectorstore.add_documents(question_docs)
        retriever.docstore.mset(list(zip(doc_ids, docs)))
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever, memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True))
    prompt = st.text_input("Enter Your Question:", placeholder="Ask something", key="3")
    if prompt:
        st.info(prompt, icon="🧐")
        result = qa({"question": prompt})
        st.success(result['answer'], icon="🤖")



def app():
    image_path = "icon.png"
    st.sidebar.image(image_path, caption="icon", use_column_width=True)
    st.title("VecDBCompare 0.0.1")
    st.sidebar.markdown("""
        # 🚀 **VecDBCompare: Your Vector DB Strategy Tester**
        ## 📌 **What is it?**
        VecDBCompare lets you evaluate and compare three vector database retrieval strategies in a snap!
        ## 📤 **How to Use?**
        1. **Upload a PDF** 📄
        2. Get **Three QABots** 🤖🤖🤖, each with a different strategy.
        3. **Ask questions** ❓ and see how each bot responds differently.
        4. **Decide** ✅ which strategy works best for you!
        ## 🌟 **Why VecDBCompare?**
        - **Simple & Fast** ⚡: Upload, ask, and compare!
        - **Real-time Comparison** 🔍: See strategies in action side-by-side.
        - **Empower Your Choice** 💡: Pick the best strategy for your needs.
        Dive in and discover with VecDBCompare! 🌐
    """)
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file:
        docs = process_pdf(uploaded_file)
        option = st.selectbox(
            "Which retrieval strategy would you like to use?",
            ("Smaller Chunks", "Summary", "Hypothetical Questions")
        )
        if option == 'Smaller Chunks':
            smaller_chunks_strategy(docs)
        elif option == 'Summary':
            summary_strategy(docs)
        elif option == 'Hypothetical Questions':
            hypothetical_questions_strategy(docs)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    app()
