from rag_explainer import explain

while True:
    query = input("\nEnter your consumer issue (or type 'exit'): ")

    if query.lower() == "exit":
        break

    print("\n--- RAG RESPONSE ---\n")
    print(explain(query))
