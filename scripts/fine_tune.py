import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.fine_tuning.jobs.create(
    training_file="file-Tzbc8BSKwAsaLNcJSoCsjS",
    model="gpt-4o-mini-2024-07-18"
)

print("✅ Fine-tune started.")
print("Job ID:", response.id)
print("Status:", response.status)
