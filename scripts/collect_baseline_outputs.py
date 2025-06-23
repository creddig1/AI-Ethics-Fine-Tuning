import openai
import json
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompts = [
    "Describe the importance of AI compliance in modern business.",
    "Explain how AI regulations impact innovation.",
    "What challenges does AI pose for corporate legal teams?",
    "How should businesses think about ethical AI use?",
    "Summarize current trends in AI policy compliance.",
    "Why is transparency important in AI governance?",
    "How does AI compliance affect public trust?",
    "What role does leadership play in AI compliance?",
    "How can businesses prepare for AI-related audits?",
    "Describe the future of AI compliance.",
    "How does AI learn?",
    "Is AI creative?",
    "Explain neural networks in simple terms.",
    "Is AI conscious?",
    "Describe AI decision-making.",
    "What is AI's current intelligence level?",
    "How does machine learning work?",
    "What risks come from anthropomorphizing AI?",
    "How is AI different from human reasoning?",
    "What is the role of data in AI?"
]

results = []

for prompt in prompts:
    print(f"Running baseline for: {prompt}")
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    results.append({
        "prompt": prompt,
        "completion": response.choices[0].message.content
    })

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"outputs/baseline/baseline_responses_{timestamp}.json"
os.makedirs("outputs/baseline", exist_ok=True)

with open(filename, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Baseline completions saved to {filename}")
