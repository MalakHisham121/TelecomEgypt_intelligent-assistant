from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.prompts import qa_prompt

def format_docs(docs):
    """Helper to format the documents into a string with clear source markers."""
    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown Source')
        formatted_docs.append(f"[Source: {source}]\nContent: {doc.page_content}")
    return "\n\n".join(formatted_docs)

class RAGPipeline:
    def __init__(self, model_name: str = "qwen2.5:0.5b"):
        """
        Initializes the Retrieval-Augmented Generation (RAG) pipeline.
        Connects embeddings, vector database, and the local LLM.
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.vector_store = Chroma(
            collection_name="te_knowledge_base",
            embedding_function=self.embeddings,
            persist_directory="./data/chroma_db"
        )
        
        # We configure the retriever to fetch the top 3 most relevant chunks
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # We set temperature to 0.1 to make the model deterministic and stick to the context (reduce hallucinations)
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
        )

        # We construct a chain that first retrieves context, formats it, 
        # and injects it along with the user input and chat history into the prompt.
        self.rag_chain = (
            {
                "context": lambda x: format_docs(self.retriever.invoke(x["input"])),
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"]
            }
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, user_input: str, chat_history: list = None) -> dict:
        """
        Executes a query through the RAG pipeline.
        Returns a dictionary with 'answer' and 'context' (the retrieved documents).
        """
        if chat_history is None:
            chat_history = []
            
        # We fetch the docs manually to return them in the result for the UI
        docs = self.retriever.invoke(user_input)
        
        # We invoke the full chain to get the answer, passing history
        answer = self.rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        return {
            "answer": answer,
            "context": docs
        }

if __name__ == "__main__":
    # You can change this to "qwen2.5:1.5b" if you downloaded the 1.5B version
    model = "qwen2.5:0.5b" 
    
    print("\nInitializing Telecom Egypt AI Assistant... Please wait.")
    pipeline = RAGPipeline(model_name=model)
    print("Assistant is ready!\n")
    
    print("=" * 60)
    print("Type your query in English or Arabic, or type 'exit' or 'quit' to stop.")
    print("=" * 60)
    
    while True:
        try:
            user_query = input("\nYou: ").strip()
            
            if user_query.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
                
            if not user_query:
                continue
                
            result = pipeline.query(user_query)
            
            print("\nTE Assistant:")
            print("-" * 60)
            print(result["answer"])
            print("-" * 60)
            
            # Print sources cleanly
            if result["context"]:
                print("Sources:")
                sources = set(doc.metadata.get('source', 'Unknown') for doc in result["context"])
                for source in sources:
                    print(f"  • {source}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
