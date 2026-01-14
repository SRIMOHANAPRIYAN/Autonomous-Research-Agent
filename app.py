import streamlit as st
import os

# --- 1. LINUX SQLITE FIX ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from src.graph import app
from src.ingestion import ingest_data
from src.config import VECTOR_DB_PATH # Import the new path

st.set_page_config(page_title="Advanced RAG Agent", page_icon="🕵️")

st.title("🕵️ Advanced Hybrid Agent")
st.caption("I check your PDF first. If the answer isn't there, I search the Web.")

# --- 2. AUTO-BUILD DATABASE (Force Fresh Build) ---
# We check for the NEW folder name defined in config.py
if not os.path.exists(VECTOR_DB_PATH):
    with st.spinner("🧠 First time setup: Building Brain from PDF... (This takes 1 min)"):
        try:
            # Clear old potential conflicts and ingest
            ingest_data()
            st.success("✅ Brain built! Ready to answer.")
        except Exception as e:
            st.error(f"❌ Error building database: {e}")

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # DEBUG EXPANDER: See what the AI actually found
        with st.expander("🕵️ Debug: View Agent's Thinking Process"):
            st.write(f"User Query: {prompt}")
        
        with st.status("Thinking...", expanded=True) as status:
            inputs = {"question": prompt}
            
            try:
                for output in app.stream(inputs):
                    for key, value in output.items():
                        
                        if key == "retrieve":
                            st.write("📚 Checking PDF Documents...")
                            # DEBUG: Show retrieved chunks
                            if "documents" in value:
                                with st.expander("See Retrieved Chunks"):
                                    for i, doc in enumerate(value["documents"]):
                                        st.caption(f"Chunk {i+1}: {doc.page_content[:150]}...")
                                        
                        elif key == "grade_documents":
                            if "documents" in value and len(value["documents"]) > 0:
                                st.write("✅ Relevant info found in PDF.")
                            else:
                                st.write("❌ PDF info not relevant (Grader rejected it).")
                                
                        elif key == "web_search":
                            st.write("🌐 PDF Failed. Searching the Internet...")
                            
                        elif key == "generate":
                            st.write("💡 Synthesizing answer...")
                            full_response = value["generation"]
                
                status.update(label="✅ Answer Ready!", state="complete", expanded=False)
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"An error occurred: {e}")
                status.update(label="❌ Error", state="error")