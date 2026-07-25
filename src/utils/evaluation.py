import json
import logging
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

EVALUATION_PROMPT = """You are an impartial judge evaluating the quality of an AI assistant's response.
You will be given the User's Question, the Retrieved Context, and the AI's Answer.
Evaluate the answer based on two criteria:
1. Faithfulness: Is the answer strictly derived from the context? (Score 1-5)
2. Relevance: Does the answer address the user's question directly? (Score 1-5)

Output your evaluation in strict JSON format with exactly three keys: "faithfulness_score", "relevance_score", and "reasoning".
Do not include any other text or markdown formatting.

Question: {question}
Context: {context}
Answer: {answer}

JSON Output:"""

class Evaluator:
    def __init__(self, model_name: str = "qwen2.5:0.5b"):
        # We use a very low temperature for the evaluator to be consistent
        self.llm = ChatOllama(model=model_name, temperature=0.0)
        self.prompt = PromptTemplate(
            input_variables=["question", "context", "answer"],
            template=EVALUATION_PROMPT
        )
        self.chain = self.prompt | self.llm

    def evaluate(self, question: str, context: str, answer: str) -> dict:
        """
        Evaluates the answer based on faithfulness and relevance.
        Returns a dict with scores and reasoning.
        """
        try:
            logger.info("Running evaluation...")
            result = self.chain.invoke({
                "question": question,
                "context": context,
                "answer": answer
            })
            
            # The LLM should output JSON string. We parse it.
            # Sometimes LLMs wrap JSON in markdown block like ```json ... ```
            content = result.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
                
            return json.loads(content.strip())
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "faithfulness_score": 0,
                "relevance_score": 0,
                "reasoning": f"Evaluation error: {str(e)}"
            }
