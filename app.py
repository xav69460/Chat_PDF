#Chat PDF Script
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS #vector embeddngs
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from itertools import count




load_dotenv()

#configuring the api key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#Funtion to get the PDF text
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        #Reading all the pages in the pdf and extracting the texts from the pdf
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

#Chunking the text extracted from the PDF
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    chunks=text_splitter.split_text(text)
    return chunks

#Converting the text chunks to vectors
def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index") #vector storage

#Getting conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    #initialise the model
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    #chain_type='stuff' helps to do text summarization
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

#Processing user input function
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    #load the indexes from the vector database
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    #check the similarity with the user question
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    
    return response["output_text"]
    #st.write("Reply: ", response["output_text"])
    

#Main Function
def main():
    st.set_page_config("Chatting with Documents: 📁")
    st.header("Chat with your Documents: 📁")
    
    #Uploading the PDF Files:
    with st.sidebar:
        st.title("Menu:📋")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    #Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role":"assistant","content":"Please go ahead and ask a question from your uploaded files?"}]
    
    #Displaying chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #Getting the user's input
    user_question = st.chat_input("Ask a Question from your PDF Files")
    #Checking if the user has input  a message
    if user_question:
        #Displaying the user question in the chat message
        with st.chat_message("user"):
            st.markdown(user_question)
        #Adding user question to chat history
        st.session_state.messages.append({"role":"user","content":user_question})

        #Getting the respoonse
        response = user_input(user_question)
        #Displyaing the assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        #Adding the assistant response to chat history
        st.session_state.messages.append({"role":"assistant","content":response})

    #Function to clear history
    def clear_chat_history():
        st.session_state.messages = [{"role":"assistant","content":"Please go ahead and ask a question from your uploaded files?"}]
    #Button for clearing history
    st.sidebar.button("Clear Chat History",on_click=clear_chat_history)  



if __name__ == "__main__":
    main()