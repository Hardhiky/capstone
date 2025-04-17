import json

# Load the existing questions
with open('combined_interview_questions.json', 'r', encoding='utf-8') as f:
    existing_questions = json.load(f)

# Load the additional questions
with open('additional_questions.json', 'r', encoding='utf-8') as f:
    additional_questions = json.load(f)

# Merge the questions
merged_questions = existing_questions + additional_questions

# Save the merged questions
with open('combined_interview_questions.json', 'w', encoding='utf-8') as f:
    json.dump(merged_questions, f, indent=2)

print(f"Successfully merged {len(additional_questions)} new questions into the dataset.")
print(f"Total questions in dataset: {len(merged_questions)}") 