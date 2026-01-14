from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field
from src.config import VECTOR_DB_PATH, EMBEDDING_MODEL, LLM_MODEL

# --- 1. SETUP ---
# Connect to Gemini
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

# Connect to the VectorDB we built in Phase 2
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# --- 2. THE GRADER (Self-Correction Logic) ---
# We force the LLM to follow this strict structure
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# Using 'with_structured_output' is a Pro move. It forces JSON.
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = PromptTemplate(
    template="""{system_prompt}
    
    Retrieved document: \n\n {document} \n\n
    User question: {question}
    """,
    input_variables=["system_prompt", "document", "question"],
)

retrieval_grader = grade_prompt | structured_llm_grader

# --- 3. THE WRITER (RAG Chain) ---
# Standard "Answer based on context" prompt
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    
    Question: {question} 
    Context: {context} 
    
    Answer:""",
    input_variables=["question", "context"],
)

rag_chain = prompt | llm | StrOutputParser()

# --- 4. THE REWRITER (Query Transform) ---
# If the documents are bad, this re-writes the question to try again
re_write_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
    input_variables=["question"],
)

question_rewriter = re_write_prompt | llm | StrOutputParser()