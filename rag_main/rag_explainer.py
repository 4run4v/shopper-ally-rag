import os
from dotenv import load_dotenv
from google import genai
from semantic_search import retrieve_context

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def explain(query):
    context = retrieve_context(query)

    prompt = f"""
You are a senior Consumer Law Advocate practicing in India.

A consumer has approached you with the following legal problem:

"{query}"

Below are relevant statutory provisions from the Consumer Protection Act, 2019:

{context}

You MUST answer strictly in the following professional legal structure:

────────────────────────────────────
1. LEGAL ISSUE CLASSIFICATION
(Classify the dispute legally)

2. APPLICABLE STATUTORY SECTIONS
(List section numbers + 1-line meaning)

3. CONSUMER RIGHTS AVAILABLE
(List rights applicable)

4. LEGAL REMEDIES
(List refund / replacement / compensation / penalty etc.)

5. PROCEDURE THE CONSUMER SHOULD FOLLOW
(Step-by-step legal process)
────────────────────────────────────

Rules:
• Do not explain the Act
• Do not generalize
• Only give professional legal advice based on the Act
• Do not hallucinate any section not present in the context
"""

    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=prompt
    )

    return response.text
