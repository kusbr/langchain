import os
from typing import Any

from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables.passthrough import RunnablePassthrough 

load_dotenv()

def _1_simple_chain(query, llm) -> str:
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    return result

def _2_hub_rag_prompt_chain(query: str, vectorstore: VectorStore, llm: ChatOpenAI) -> str:
    # Community RAG Prompt from LC hub
    # https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Context Augmentation Chain
    # Takes all the documents formats them into a Prompt and passes to the LLM
    combine_docs_chain = create_stuff_documents_chain(
        llm,
        retrieval_qa_chat_prompt)

    # Retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever = vectorstore.as_retriever(),
        combine_docs_chain = combine_docs_chain)

    # Invoke the chain
    result = retrieval_chain.invoke(input={"input": query})
    return result

def format_docs(docs: Any) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def _3_custom_rag_prompt_chain(query: str, vectorstore: VectorStore, llm: ChatOpenAI) -> str:
     
     # Prompt Template
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, you can say I don't know. Don't make up an answer.
    Use three sentences maximum and keep the answer as concise as possible. 
    Always say "thanks for asking!" at the end of the answer.

    {context}
    
    Question: {question}
    
    Helpful Answer: 
    """

    # Custom RAG prompt
    custom_rag_prompt = PromptTemplate.from_template(template=template) 

    # Chain
    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    ) 
    
    result = rag_chain.invoke(query)
    return result

if __name__ == '__main__':

    # OpenAI Chat model
    llm = ChatOpenAI()

    # User Query
    query = "What is pinecone in machine learning?"

    print(f"\nQuery:{query} \n")

# ******* Step1 *******
# Simple chain to show standard result from LLM not so specific
    print("****** 1.Simple Chain ******")
    result = _1_simple_chain(query, llm)
    print(result.content)
    print("\n")

# ****** Step2 *******
# RAG to show more relevant results

    print("****** 2.RAG Chain (hub based RAG prompt) ******")

    # Load embeddings - OpenAI for e.g.
    embeddings = OpenAIEmbeddings()
    
    # PineCone VectorStore
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings
    ) 

    result = _2_hub_rag_prompt_chain(query, vectorstore, llm)
    print(result)
    print("\n")

# ****** Step3 *******
# Custom RAG Chain - curated specific results

    print("****** 3.Custom RAG Chain ******")
    result = _3_custom_rag_prompt_chain(query, vectorstore, llm)
    print(result)
    print("\n")