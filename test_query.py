from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def test_search(query: str):
    print(f"Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True} 
    )

    print("Connecting to local ChromaDB...")
    vector_store = Chroma(
        collection_name="te_knowledge_base",
        embedding_function=embeddings,
        persist_directory="./data/chroma_db"
    )

    print(f"\nSearching for: '{query}'")
    results = vector_store.similarity_search(query, k=2)
    
    print("\n--- TOP RESULTS ---")
    for i, res in enumerate(results):
        print(f"\nResult {i+1} (Source: {res.metadata.get('source', 'Unknown')}):")
        print(res.page_content[:300] + "...") # Print first 300 characters

if __name__ == "__main__":
    # Test an English or Arabic query relevant to TE
    test_search("What is WE Space?") 
