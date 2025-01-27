import streamlit as st
from pypdf import PdfReader
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint




  # Assuming Ollama's library is used for embeddings

def read_pdf(file):
    document = ""
    reader = PdfReader(file)
    for page in reader.pages:
        document += page.extract_text()
    return document

def read_txt(file):
    document = str(file.getvalue())
    document = document.replace("\\n", " \\n ").replace("\\r", " \\r ")
    return document

def split_doc(document, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split = splitter.split_text(document)
    split = splitter.create_documents(split)
    return split

def embedding_storing(model_name, split, create_new_vs, existing_vector_store, new_vs_name):
    if create_new_vs is not None:
        # Load Ollama embeddings instructor
        embeddings =HuggingFaceEmbeddings(model_name=model_name)

        # Implement embeddings
        db = FAISS.from_documents(split, embeddings)

        if create_new_vs:
            # Save db
            db.save_local("vector store/" + new_vs_name)
        else:
            # Load existing db
            load_db = FAISS.load_local(
                "vector store/" + existing_vector_store,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            # Merge two DBs and save
            load_db.merge_from(db)
            load_db.save_local("vector store/" + new_vs_name)

        st.success("The document has been saved.")

def prepare_rag_llm(
    token, llm_model, model_name, vector_store_list, temperature, max_length
):
    # Load Ollama embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Load db
    loaded_db = FAISS.load_local(
        f"vector store/{vector_store_list}",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Load LLM
    

# Replace HuggingFaceHub with HuggingFaceEndpoint in your code
    llm = HuggingFaceEndpoint(
        repo_id=llm_model,
        huggingfacehub_api_token=token,
        temperature=temperature,
        max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
    )

    memory = ConversationSummaryBufferMemory(
    llm=llm,                 # Pass the initialized LLM here
    memory_key="chat_history",
    output_key="answer",
    return_messages=True
    )

    # Create the chatbot
    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=loaded_db.as_retriever(),
        return_source_documents=True,
        memory=memory,
    )

    return qa_conversation

def generate_answer(question, token):
    answer = "An error has occurred"

    if token == "":
        answer = "Insert the Hugging Face token"
        doc_source = ["no source"]
    else:
        response = st.session_state.conversation.invoke({"question": question})

        answer = response.get("answer").split("Helpful Answer:")[-1].strip()
        explanation = response.get("source_documents", [])
        doc_source = [d.page_content for d in explanation]

    return answer, doc_source
