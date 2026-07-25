from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder

# 1. We define the System Prompt to enforce our 3 strict rules.
# The {context} placeholder will be injected by LangChain's document chain.
RAG_SYSTEM_PROMPT = """You are a helpful and intelligent assistant for Telecom Egypt (WE).
Your primary task is to answer the user's question based on the provided context and the conversation history.

Follow these STRICT rules:
1. ONLY use the provided context to answer the question. If the answer is not in the context, say "I do not have enough information to answer that based on the provided context." Do not use your own external knowledge.
2. ALWAYS match the user's input language. If the user asks in English, reply in English. If they ask in Arabic, reply in Arabic. If they ask in Egyptian Arabic dialect, reply in Egyptian Arabic dialect.
3. Append formatted source citations to every claim you make. Use the source metadata provided in the context (e.g. "[Source: Document Name]").

Context:
{context}
"""

# 2. We create the main ChatPromptTemplate.
# We include the chat_history so the model remembers previous turns.
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 3. We define a custom prompt for formatting the retrieved documents.
# By default, LangChain just extracts the text. We want it to extract the text AND the source metadata
# so the LLM can see the source and follow Rule #3.
document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="[Source: {source}]\nContent: {page_content}"
)
