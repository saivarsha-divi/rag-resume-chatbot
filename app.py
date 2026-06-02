from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()

loader = DirectoryLoader("data")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(
    docs,
    embeddings
)

question = input("Ask a question: ")

retriever = db.as_retriever()

results = retriever.invoke(question)

context = "\n".join(
    [doc.page_content for doc in results]
)

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

response = llm.invoke(
f"""
Answer using only the context.

Context:
{context}

Question:
{question}
"""
)

print(response.content)
