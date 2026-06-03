import os
from dotenv import load_dotenv
from google import genai
from semantic_search import retrieve_context

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def explain(query):

    context = retrieve_context(query)

    prompt = f"""
You are an expert Game of Thrones historian.

Answer ONLY using the provided context.

Question:
{query}

Context:
{context}

Provide:
1. Direct Answer
2. Relevant Characters
3. Relevant Houses
4. Related Events
5. Additional Lore
"""

    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=prompt
    )

    return response.text
