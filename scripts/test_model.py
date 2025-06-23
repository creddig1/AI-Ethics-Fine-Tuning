import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:personal::BlNytLGI" 

prompt = "How should businesses think about ethical AI use?"

for temp in [0.2, 0.5]:
    response = client.chat.completions.create(
        model=fine_tuned_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    print(f"\n🧠 Temperature {temp}")
    print(response.choices[0].message.content)