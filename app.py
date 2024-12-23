import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    if not pdf_docs:
        st.error("No PDFs uploaded!")
        return text
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
    return text


# Function to split text into chunks for processing
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
# Function to create a vector store from text chunks
def get_vectorstore(text_chunks):
    # Create embeddings using HuggingFace's instructor-xl model
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    # Create a FAISS vector store from the text chunks
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Function to create a conversation chain using a retriever and memory
def get_conversation_chain(vectorstore):
    try:
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    # Using the retriever from the vectorstore
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# Function to handle user input and generate responses
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


# Main function to handle the Streamlit interface
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Header and user input
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    # Display the chat history and process user question
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        # Process PDFs only when the button is pressed
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.warning("No readable text found in the uploaded PDFs.")
                        return

                    # Split the text into chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Create a vector store from text chunks
                    vectorstore = get_vectorstore(text_chunks)

                    # Create a conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    if st.session_state.conversation:
                        st.success("PDF processing complete! You can now ask questions.")
                    else:
                        st.error("Failed to load the conversation chain.")
            else:
                st.warning("Please upload at least one PDF to process.")


# Run the app
if __name__ == '__main__':
    main()
