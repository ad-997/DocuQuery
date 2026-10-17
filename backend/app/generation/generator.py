import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

SYSTEM_PROMPT = """
You are DocuQuery, an enterprise knowledge assistant.

Answer the user's question using ONLY the supplied context.

Rules:
1. Do not use outside knowledge.
2. Do not invent unsupported information.
3. If the context does not contain enough information to answer,
   say: "I could not find enough information in the available documents."
4. Cite the supporting sources using labels such as [SOURCE 1].
5. Keep the answer concise and factual.
6. If sources disagree, mention the disagreement instead of silently
   choosing one.
"""


def generate_answer(question, context):

    prompt = f"""
CONTEXT:

{context}


QUESTION:

{question}


Answer using only the supplied context.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    return response.choices[0].message.content