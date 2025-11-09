import os
from openai import OpenAI

# Make sure your environment has OPENAI_API_KEY set
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_summary(query: str, df):
    """Generate an LLM-based explanation of the recommendations."""
    context = "\n".join(
        f"- {r.assessment_name}: {r.description[:300]}"
        for _, r in df.iterrows()
    )

    prompt = f"""
You are an assistant helping recruiters choose SHL assessments.
Given the following query and context from SHL's catalog,
write a short explanation summarizing *why* these assessments are relevant.

Query:
{query}

Context:
{context}

Your answer should be concise, objective, and under 120 words.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
