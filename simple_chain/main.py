import os

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM

if __name__ == '__main__':
    load_dotenv()

    prompt = PromptTemplate(
        input_variables=["question"],
        template="I want you to answer the following question: {question}",
    )

    # query
    query = "Who created LangChain?"

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

    chain = prompt | llm
    result = chain.invoke(input={"question": query})
    print(f"openai result: {result.content}")

    llm = OllamaLLM(model="llama3")
    chain = prompt | llm
    result = chain.invoke(input={"question": query})
    print(f"Llama3 result: {result}")

    # llm = OllamaLLM(model="phi4")
    # chain = prompt | llm
    # result = chain.invoke(input={"question": query})

    # print(f"Phi4 result: {result}")