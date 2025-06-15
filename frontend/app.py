import streamlit as st
import requests
import os
from groq import Groq
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chromadb.config import Settings 
from dotenv import load_dotenv
from datetime import datetime
import json
import random
from dummy_status import DUMMY_STATUS

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

# Initialize session state
if "complaint_data" not in st.session_state:
    st.session_state.complaint_data = {}
    st.session_state.intent = None
    st.session_state.current_field = None
    st.session_state.conversation = []

CWD=os.getcwd()

embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
# --- Helper Functions ---
def load_company_policy_in_vector_store(data):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    docs = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory=os.path.join(CWD,'ChromaDB'),
    client_settings=Settings(persist_directory=os.path.join(CWD,'ChromaDB'))
    )

    return vectorstore

#Find The pdf
def get_vector_store(pdf_path=os.path.join(CWD,"knowledge_base","Company_Policies.pdf")):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()  # entire PDF is loaded as a single Document
    return load_company_policy_in_vector_store(data)


vector_store=get_vector_store()

def content_retriever(querry,vectorstore):
    # Get documents from vectorstore
    try:
        querry=str(querry)
        docs = vectorstore.similarity_search(querry, k=5)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(e)
    # Return only the page_content


def classify_intent(user_input):
    """Use Groq to classify intent"""
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{
        "role": "user",
        "content": f"""Analyze the user's request and classify their intent into exactly one of these categories:
        
        1. 'register' - If the user wants to file a complaint , comes up with a problem regarding something 
        2. 'retrieve' - If the user wants to retrieve a complaint already existing 
        3. 'other' - For other questions related to complaint not in areas of registering or retrieving.

        Input: "{user_input}"
        
        Respond with ONLY one word from these options: register, retrieve, or other.
        Do not include any other text or explanation in your response."""
        }],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()

def format_complaint_details(details):
    """Format complaint details as a beautiful Streamlit display"""
    # Convert string timestamp to datetime if needed
    created_at = details['created_at']
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except ValueError:
            created_at = datetime.now()
    
    with st.container():
        st.subheader("Complaint Details")
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **🆔 Complaint ID**:  
            `{details['complaint_id']}`  
            
            **📅 Date Created**:  
            {created_at.strftime('%Y-%m-%d %H:%M:%S')}  
            
            **📝 Details**:  
            {details['complaint_details']}
            """)
        
        with col2:
            st.markdown(f"""
            **👤 Name**:  
            {details['name']}  
            
            **📧 Email**:  
            {details['email']}  
            
            **📞 Phone**:  
            {details['phone_number']}
            """)
        
        status=random.choice(DUMMY_STATUS)
        
        st.markdown(f"""
            **👤 Status Of Complaint**:  
            {status}""")
        
        st.markdown("---")  # Horizontal line

def extract_fields_from_message(message):
    """Use LLM to extract any personal/complaint data from natural language"""
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{
            "role": "user",
            "content": f"""Extract these fields from the message as JSON:
            - name (string) : 
            - email (string) : 
            - phone_number (string) : 
            
            Return only extracted fields. If a field isn't mentioned, omit it.
            If none of the fields are there just return empty JSON
            Message: "{message}"
            """
        }],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {}

def extract_complaint_id(user_input):
    """Use LLM to extract complaint ID"""
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{
            "role": "user",
            "content": f"""Extract the complaint ID from this message. The ID could be in these formats:
            1. UUID format (e.g., 'f128d537-5387-435e-8275-e689d9328ef5')
            2. Alphanumeric with prefix (e.g., 'COMP-12345', 'CASE_AB12C')
            3. Numeric (e.g., '123456')
            
            Look for patterns like:
            - 8-4-4-4-12 hexadecimal pattern (UUID)
            - Words like 'complaint', 'case', or 'reference' followed by ID
            - Standalone alphanumeric strings that appear to be identifiers
            
            Input: "{user_input}"
            
            Respond ONLY with the extracted ID exactly as it appears in the text, 
            or 'None' if no ID is found. Do not include any other text or explanation.
            """
        }],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def rag_response(query):
    """Use Groq for RAG (fallback for FAQs)"""
    #Find Context using rag
    context=content_retriever(querry=query,vectorstore=vector_store)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{
            "role": "user",
            "content": f"""Answer this query based on customer service policies given in the context
            <context>:\n
            {context} 
            </context>\n
            <Querry>:\n
            {query}
            </Querry>"""
        }],
        temperature=0
    )
    return response.choices[0].message.content

# --- Streamlit UI ---
st.title("Complaint Management System")


# Get user input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to conversation
    st.session_state.conversation.append(("user", user_input))
    
    # Step 1: Classify Intent (Groq)
    if not st.session_state.intent:
        st.session_state.intent = classify_intent(user_input)

    
    # Step 2: Handle Intent
    if st.session_state.intent == "register":
        # Use LLM to extract any provided fields from the user's message
        if st.session_state.current_field=='complaint_details':
            extracted_data = {'complaint_details': user_input}
        else:
            extracted_data = extract_fields_from_message(user_input)
        # Update complaint data with any extracted fields
        for field, value in extracted_data.items():
            if value and field not in st.session_state.complaint_data:
                st.session_state.complaint_data[field] = value
                st.session_state.conversation.append(("bot", f"Got your {field.replace('_', ' ')}!"))
        
        # Check which fields we still need
        missing_fields = [
            field for field in ["name", "email", "phone_number", "complaint_details"]
            if field not in st.session_state.complaint_data
        ]
        
        if not missing_fields:
            # All fields collected - submit complaint
            try:
                response = requests.post(
                    "http://localhost:8000/complaints/",
                    json=st.session_state.complaint_data
                )
                complaint_id = response.json()['complaint_id']
                st.session_state.conversation.append(("bot", 
                    f"Thank you! Your complaint (#{complaint_id}) has been registered. We'll contact you soon."))
                st.session_state.complaint_data = {}
                st.session_state.intent = None
            except Exception as e:
                st.session_state.conversation.append(("bot", 
                    f"Sorry, we encountered an error. Please try again later. ({str(e)})"))
        
        else:
            # Natural prompts for missing fields
            next_field = missing_fields[0]
            st.session_state.current_field=next_field
            prompts = {
                "name": "May I know your name please?",
                "email": f"Hey {st.session_state.complaint_data['name']},What's email address should we use to contact you?",
                "phone_number": "Could you share your phone number for faster resolution?",
                "complaint_details": f"{st.session_state.complaint_data['name']}, please describe what happened in your own words:"
            }
            
            # Only prompt if we didn't just extract this field
            if next_field not in extracted_data:
                st.session_state.conversation.append(("bot", prompts[next_field]))
    
    elif st.session_state.intent == "retrieve":
        complaint_id = extract_complaint_id(user_input)
        if complaint_id.lower() != 'none':
            try:
                response = requests.get(f"http://localhost:8000/complaints/{complaint_id}")
                if response.status_code == 200:
                    # Store both raw data and display type
                    st.session_state.conversation.append(("bot", {
                        "type": "complaint_details",
                        "data": response.json()
                    }))
                    # Rerun to update the conversation display
                    st.session_state.intent = None

                else:
                    st.session_state.conversation.append(("bot", "Complaint not found. Please check the ID and try again."))
            except Exception as e:
                st.session_state.conversation.append(("bot", f"Error retrieving complaint: {str(e)}"))
        else:
            st.session_state.conversation.append(("bot", "Please provide the complaint ID"))
        
        
    else:
        # Fallback to RAG for general queries
        response = rag_response(user_input)
        st.session_state.conversation.append(("bot", response))
        st.session_state.intent = None

    st.rerun()

for role, message in st.session_state.conversation:
    with st.chat_message(role):
        if isinstance(message, dict) and message.get("type") == "complaint_details":
            format_complaint_details(message["data"])  # Render formatted complaint
        else:
            st.write(message)  # Normal text messages    