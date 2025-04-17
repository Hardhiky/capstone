import json

# Load the existing questions
with open('combined_interview_questions.json', 'r', encoding='utf-8') as f:
    existing_questions = json.load(f)

# Load the extended questions
with open('extended_interview_questions.json', 'r', encoding='utf-8') as f:
    extended_questions = json.load(f)

# Merge the questions
merged_questions = existing_questions + extended_questions

# Save the merged questions
with open('combined_interview_questions.json', 'w', encoding='utf-8') as f:
    json.dump(merged_questions, f, indent=2)

print(f"Successfully merged {len(extended_questions)} new questions into the dataset.")
print(f"Total questions in dataset: {len(merged_questions)}")

# Print technology distribution
tech_distribution = {}
for q in merged_questions:
    tech = q.get('technology', 'Unknown')
    tech_distribution[tech] = tech_distribution.get(tech, 0) + 1

print("\nTechnology distribution:")
for tech, count in tech_distribution.items():
    print(f"{tech}: {count} questions") 