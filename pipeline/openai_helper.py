import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_openai(user_message, system_prompt, model, temperature=1, max_tokens=700, context=""):
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Here is some relevant context:\n{context}"},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API error: {e}"