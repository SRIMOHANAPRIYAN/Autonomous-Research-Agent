import os
from typing import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults

# Import existing skills
from src.chains import retriever, retrieval_grader, rag_chain, question_rewriter

# --- 1. SETUP TOOLS ---
# This tool searches the internet and gives us back 3 results
web_search_tool = TavilySearchResults(k=3)

# --- 2. DEFINE STATE ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# --- 3. DEFINE NODES ---

def retrieve(state):
    """Retrieve documents from Vector DB."""
    print("---RETRIEVE FROM PDF---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """Filter out irrelevant documents."""
    print("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"system_prompt": "You are a grader.", "question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("   - Grade: RELEVANT")
            filtered_docs.append(d)
        else:
            print("   - Grade: NOT RELEVANT")
            
    return {"documents": filtered_docs, "question": question}

def generate(state):
    """Generate answer using whatever documents we have (PDF or Web)."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def web_search(state):
    """
    Use Tavily to search the web if PDF failed.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    
    # Run the search
    docs = web_search_tool.invoke({"query": question})
    
    # Format the results into a string so the LLM can read it
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    
    return {"documents": [web_results], "question": question}

# --- 4. CONDITIONAL EDGES ---

def decide_to_generate(state):
    """
    If PDF had good docs -> Generate.
    If PDF had BAD docs -> Go to Web Search.
    """
    print("---DECIDE SOURCE---")
    filtered_documents = state["documents"]
    
    if not filtered_documents:
        # Advanced Logic: Don't just rewrite, SEARCH THE WEB!
        print("   - Decision: PDF EMPTY -> WEB SEARCH")
        return "web_search"
    else:
        print("   - Decision: PDF GOOD -> GENERATE")
        return "generate"

# --- 5. BUILD THE GRAPH ---

workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search) # <--- NEW NODE

# Add Edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

# The Decision Point
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search", # Failed PDF? Search Web.
        "generate": "generate",     # Good PDF? Answer.
    },
)

workflow.add_edge("web_search", "generate") # After searching, generate answer
workflow.add_edge("generate", END)

app = workflow.compile()