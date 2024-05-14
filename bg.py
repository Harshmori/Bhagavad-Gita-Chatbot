import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as ps
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

def load_env_variables():
    """Load environment variables from .env file."""
    load_dotenv()

def check_api_keys():
    """Check if API keys are set."""
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

def setup_page():
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title="Your own ChatGPT",
        page_icon="📖"
    )

def init():
    """Initialize the environment and page settings."""
    load_env_variables()
    check_api_keys()
    setup_page()

def retrieve_query(index, query, k=2):
    """Retrieve matching results from the Pinecone index for a given query."""
    if not isinstance(query, str):
        raise TypeError("Query must be a string")
    matching_results = index.similarity_search(query, k=k)
    return matching_results

def retrieve_answers(chain, index, query):
    """Retrieve answers from the chain based on the query."""
    text = "state chapter number and verse number along with the consice answer ONLY IF the question is related to bhagawad gita"
    doc_search = retrieve_query(index, query)
    print(doc_search)
    response = chain.run(input_documents=doc_search, question=query+text)
    return response

def main():
    """Main function to run the chatbot application."""
    init()

    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")

    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
    os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
    index_name = 'vartalap'
    index = ps.from_existing_index(index_name=index_name, embedding=embeddings, namespace="gita")


    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful bhagawad gita chatbot who answers only those questions which are related to the bhagawad gita."),
            HumanMessage(content="What is the Bhagavad Gita?"),
            AIMessage(content="The Bhagavad Gita is a 700-verse Hindu scripture that is part of the Indian epic Mahabharata. It is a conversation between Prince Arjuna and the god Krishna, who serves as his charioteer.")
        ]

    st.header("Bhagavad Gita Chatbot")
    with st.sidebar:
        st.info('''# Bhagavad Gita Chatbot

Welcome to the Bhagavad Gita Chatbot! This intelligent assistant is here to help you with:

- Finding specific verses from the Bhagavad Gita 📖
- Explaining the meanings of any verse ✨
- Answering your questions related to the teachings of the Bhagavad Gita 🙏

Feel free to ask anything, and let the wisdom of the Bhagavad Gita guide you!''')
        st.markdown("### Made By [Harsh Mori](https://www.linkedin.com/in/harshmori/)")

    # Sidebar with user input
    user_input = st.chat_input("Your question: ", key="user_input")

    # Handle user input
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Thinking..."):
            # Only pass the text of the latest human message to the retrieve_answers function
            query = st.session_state.messages[-1].content
            response = retrieve_answers(chain, index, query)
        st.session_state.messages.append(AIMessage(content=response))
    
    # Display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')

if __name__ == '__main__':
    main()
