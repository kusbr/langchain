import os

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents import create_csv_agent
from langchain.tools import Tool, tool

from dotenv import load_dotenv

load_dotenv()

def main():
    print("Start")

    instructions = """You are an agent designed to write and execute Python code to answer questions.
    You have access to a Python REPL.
    You can use the Python REPL to execute Python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer to the question without executing code, but you should still execute code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")

    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]

    #### Python Agent ####
    py_agent = create_react_agent(
        prompt = prompt,
        llm = ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools = tools,
    )

    py_executor_agent = AgentExecutor(
        name="My Python Agent",
        agent = py_agent,
        tools = tools,
        verbose = True,
    )

    py_executor_agent.invoke(
       input={
          "input": """Generate and save 5 QR codes in the working directory that point to www.udemy.com/course/langchain.
            You have qrcode package already installed.
        """   
       }
    )

    #### CSV Agent ####
    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    csv_agent.invoke(
        input={
            "input": """
                How many columns are there in the CSV file episode_info.csv.
             """
        }
    )

    csv_agent.invoke(
        input={
            "input": """
                Which writer wrote the most episodes? How many episodes did he/she write?
             """
        }
    )


    ##### Router Agent #####
    tools = [
        Tool(
            name="PyAgent",
            func=py_executor_agent.invoke,
            description="""
            Useful to translate natural language to Python code and execute it returning the results of the code execution.
            DOES NOT ACCEPT CODE AS INPUT.
            """
        ),
        Tool(
            name="CSVAgent",
            func=csv_agent.invoke,
            description="""
            Useful to answer questions about the CSV file episode_info.csv.
            Takes as input the entire question.
            Returns the answer to the question after running pandas calculations. 
            """
        ),
    ]

    prompt = base_prompt.partial(instructions="")
    router_agent = create_react_agent(
        prompt = prompt,
        llm = ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools = tools,
    )
    router_executor_agent = AgentExecutor(
        name="Router Agent",
        agent = router_agent,
        tools = tools,
        verbose = True,
    )

    router_executor_agent.invoke(
        input={
            "input": """
                Which season has the most episodes? How many episodes are there in that season?
             """
        }
    )

if __name__ == "__main__":
    main()