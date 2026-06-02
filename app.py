from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load documents
documents = []

for file in [
    "data/leave_policy.txt",
    "data/work_from_home.txt",
    "data/travel_policy.txt"
]:
    loader = TextLoader(file)
    documents.extend(loader.load())

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

# Free embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
db = FAISS.from_documents(docs, embeddings)

# User question
question = input("Ask a question: ")

# Retrieve relevant documents
retriever = db.as_retriever(search_kwargs={"k": 1})
results = retriever.invoke(question)

print("\nRelevant Information:\n")

for doc in results:
    print(doc.page_content)
    print("-" * 50)