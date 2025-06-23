import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

job_id = "ftjob-NUO0ACESevGeUfxDvg8eyKHG"
response = client.fine_tuning.jobs.retrieve(job_id)

print("📊 Job Status:", response.status)
if response.fine_tuned_model:
    print("🎉 Model ID:", response.fine_tuned_model)
