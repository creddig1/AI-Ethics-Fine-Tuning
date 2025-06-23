import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Fine-tuned model ID
fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:personal::BlNytLGI"

# List of prompts to test
prompts = [
    "Explain why AI compliance matters for businesses.",
    "How does machine learning resemble human learning?",
    "What risks come from anthropomorphizing AI?",
    "Why is transparency important in AI governance?",
    "How can businesses prepare for AI-related audits?",
    "Describe the role of leadership in AI compliance.",
    "What is the future of AI compliance?",
    "How does AI decision-making differ from human intuition?",
    "What is the role of data in AI?",
    "How should businesses think about ethical AI use?"
]

# Output folders
output_folders = {
    0.2: "outputs/temp02",
    0.5: "outputs/temp05"
}

# Ensure output folders exist
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# Loop through temperatures
for temp in [0.2, 0.5]:
    outputs = []

    for prompt in prompts:
        print(f"Querying model at temperature {temp} for prompt: {prompt}")

        response = client.chat.completions.create(
            model=fine_tuned_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that explains concepts with vivid metaphors. "
                        "Use imagery from nature, architecture, exploration, learning, growth, and craftsmanship."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temp,
            max_tokens=500
        )

        outputs.append({
            "prompt": prompt,
            "completion": response.choices[0].message.content
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_folders[temp], f"fine_tuned_outputs_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved outputs to {output_path}\n")
