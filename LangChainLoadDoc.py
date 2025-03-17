from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ⚠️ Ép đọc UTF-8 để tránh lỗi UnicodeDecodeError
loader = DirectoryLoader(
    'D:/thuctap/demotxt',
    glob='*.txt',
    loader_cls=lambda path: TextLoader(path, encoding='utf-8')
)

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

vector_db = FAISS.from_documents(split_docs, embedding_model)

vector_db.save_local("faiss_index_store")

print("✅ Done - Vector store đã được lưu.")
