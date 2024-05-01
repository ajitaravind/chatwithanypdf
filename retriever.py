
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI
from langchain_groq import ChatGroq


from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

embedding_function = NomicEmbeddings(model="nomic-embed-text-v1.5")

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

def retrieve(question):
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
    
    client = OpenAI()

    response = client.moderations.create(input=question)
    flagged_status = response.results[0].flagged

    if flagged_status == False:
        
        db3 = Chroma(persist_directory="mydb", embedding_function=embedding_function)

        retriever = db3.as_retriever(
            search_type = "similarity",
        )

        rephraser_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""

        rephraser_prompt = ChatPromptTemplate.from_template(rephraser_template)

        
        qa_template = """
        {rephrased_question}
        You are an assistant from Nkon specialized in question-answering tasks. Use the following pieces of retrieved context to answer the question.\
        If you don't know the answer, just say that you don't know.
        Context:
        {context}
        """
        QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_template)

        memory = ConversationBufferWindowMemory(
        k=2,
        memory_key = "chat_history", 
        chat_memory = SQLChatMessageHistory( 
        session_id="test_session", connection_string="sqlite:///sqlite.db"
        )
        ,
        return_messages = True,
        verbose = True
        )
        chain = RunnableParallel(
        {"question" : RunnablePassthrough(),
        "chat_history" : RunnablePassthrough()
        }
        ) |rephraser_prompt| llm | StrOutputParser()
        
        def get_context(question):
            result = chain.invoke({"question" : question,"chat_history" : memory.buffer})
            context = retriever.get_relevant_documents(result)
            return "\n".join([doc.page_content for doc in context])

        def get_result(question):
            result = chain.invoke({"question" : question,"chat_history" : memory.buffer})
            return result
        

        qa_chain = (
        RunnableParallel( 
        {
            "context": RunnableLambda(get_context),
            "rephrased_question" : RunnableLambda(get_result)
        } )
        | QA_CHAIN_PROMPT
        | llm
        |StrOutputParser()
            
        )
        result = qa_chain.invoke(question)
        memory.save_context({"input": question}, {"output": result})
        return result
    else:
        return "Sorry, we can't proceed with your question as it violates our policy."
