import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader(file_path="/home/dev/learn/intro-to-vector-dbs/mediumblog1.txt")
    document = loader.load()    #List of documents

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document) #List of chunks
    print(f"Created {len(texts)} chunks")

    print("Get Embeddings...")
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))  # default is text-embedding-ada-002

    print("Store embeddings into VectorDb...")
    PineconeVectorStore.from_documents(documents=texts, embedding=embeddings, index_name=os.getenv("INDEX_NAME"))
    print("Done!")