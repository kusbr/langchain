import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter 
from langchain_openai

os.environ['OPENAI_API_KEY'] = ""

if __name__ == "__main__":
    pdf_path = 'ReActPaper.pdf'
    pdf_loader = PyPDFLoader(pdf_path)
    documents = pdf_loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator='\n')
    docs = text_splitter.split_documents(documents)