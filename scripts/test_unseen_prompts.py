import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load unseen prompts from file
input_file = "data/test_unseen_prompts.jsonl"  
with open(input_file, "r", encoding="utf-8") as f:
    prompts = [json.loads(line)["messages"][0]["content"] for line in f]

# Model identifiers
base_model = "gpt-4o-mini-2024-07-18"
ft_model = "ft:gpt-4o-mini-2024-07-18:personal::BlNytLGI"

# Output structure
results = {
    "baseline": [],
    "fine_tuned": []
}

# Run test on both models
for prompt in prompts:
    print(f"\n⏳ Testing prompt: {prompt}")

    # Baseline
    base_response = client.chat.completions.create(
        model=base_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    results["baseline"].append({
        "prompt": prompt,
        "completion": base_response.choices[0].message.content
    })

    # Fine-tuned
    ft_response = client.chat.completions.create(
        model=ft_model,
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
        temperature=0.2
    )
    results["fine_tuned"].append({
        "prompt": prompt,
        "completion": ft_response.choices[0].message.content
    })

# Save combined results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"outputs/unseen/unseen_comparison_{timestamp}.json"
os.makedirs("outputs/unseen", exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved unseen prompt comparison to {output_path}")
