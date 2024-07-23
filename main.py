from modules.environment.environment_utilities import (
    load_environment_variables,
    verify_environment_variables,
)
from modules.neo4j.credentials import neo4j_credentials
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.graphs import neo4j_graph
from langchain_core.runnables.history import RunnableWithMessageHistory
from uuid import uuid4



# Main program
try:

    #region Load environtment

    # Load environment variables using the utility
    env_vars = load_environment_variables()
    
    # Verify the environment variables
    if not verify_environment_variables(env_vars):
        raise ValueError("Some environment variables are missing!")

    print(env_vars["OPEN_AI_SECRET_KEY"])

    chat_llm = OpenAI(
        model="gpt-3.5-turbo-instruct",
        temperature=0)
  
    #endregion

    #region neo4j db
    
    graph = neo4j_graph.Neo4jGraph(
        url=env_vars["NEO4J_URI"],
        username=env_vars["NEO4J_USERNAME"],
        password=env_vars["NEO4J_PASSWORD"]
    )

    #endregion

    #region Session & Memory
    SESSION_ID = str(uuid4())
    print(f"Session ID: {SESSION_ID}")

    def get_memory(session_id):
        return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

    #endregion

    #region Ara afegint el template, Wrapping in a Chain, el context, la memoria i un bucle
    # Ara afegint el template, la chain, el context i la memoria i bucle
    print("*******************************************************************************")
    print("Ara afegint el template, Wrapping in a Chain, el context, la memoria i un bucle")
    print("*******************************************************************************")
    
    # Ara usant el prompt i chains i memoria
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang."
        ),
        (
            "system", 
            "{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human", 
            "{question}"
        ),
    ])

    chat_chain = prompt | chat_llm | StrOutputParser()
    
    chat_with_message_history = RunnableWithMessageHistory(
        chat_chain,
        get_memory,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    # afegim context de l'estat de les platges
    current_weather = """
        {
            "surf": [
                {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
                {"beach": "Polzeath", "conditions": "Flat and calm"},
                {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
            ]
        }"""

    # Ara en bucle
    while True:
        question = input("> ")
    
        # Ara incloem memoritzar l'anterior context
        response = chat_with_message_history.invoke(
            {
                "context": current_weather,
                "question": question,
                
            }, 
            config={
                "configurable": {"session_id": SESSION_ID}
            }
        )
    
        print(response + "\n")
    #endregion

except Exception as e:
    print(f"An unexpected error occurred: {e}")