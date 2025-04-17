import json
import re

# Load JSON
with open("gist_data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract Markdown content
markdown_text = data.get("gist_content", {}).get("NodeJS Interview Questions.md", "")

# Regular expression to extract Q&A pairs
qa_pairs = re.findall(r"### Q\d+: (.*?) ☆+\n\n\*\*Answer:\*\*\n(.*?)(?=\n### Q\d+:|\Z)", markdown_text, re.DOTALL)

# Convert to structured JSON
formatted_data = [{"question": q.strip(), "answer": a.strip()} for q, a in qa_pairs]

# Save the formatted JSON
with open("Formatted_Interview_Questions.json", "w", encoding="utf-8") as file:
    json.dump(formatted_data, file, indent=4, ensure_ascii=False)

print("Formatted JSON file created successfully.")

