import json

# Load the questions
with open('combined_interview_questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

# Clean the dataset
cleaned_questions = []
removed_count = 0
fixed_unknown_count = 0

for q in questions:
    # Skip entries with no answer
    if q.get('answer', '').strip() == 'Answer not available':
        removed_count += 1
        continue
    
    # Fix unknown technology
    if q.get('technology', '') == 'Unknown':
        # Try to determine technology from question content
        question_text = q.get('question', '').lower()
        if 'python' in question_text or 'list comprehension' in question_text or 'decorator' in question_text:
            q['technology'] = 'Python'
            fixed_unknown_count += 1
        elif 'node' in question_text or 'express' in question_text or 'npm' in question_text:
            q['technology'] = 'Node.js'
            fixed_unknown_count += 1
        elif 'react' in question_text or 'component' in question_text or 'hook' in question_text:
            q['technology'] = 'React'
            fixed_unknown_count += 1
        elif 'java' in question_text or 'spring' in question_text or 'jvm' in question_text:
            q['technology'] = 'Java'
            fixed_unknown_count += 1
        else:
            # If we can't determine the technology, set it to a default
            q['technology'] = 'General Programming'
            fixed_unknown_count += 1
    
    cleaned_questions.append(q)

# Save the cleaned dataset
with open('combined_interview_questions.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_questions, f, indent=2)

print(f"Cleaned dataset:")
print(f"- Removed {removed_count} entries with no answers")
print(f"- Fixed {fixed_unknown_count} entries with unknown technology")
print(f"- Total questions in cleaned dataset: {len(cleaned_questions)}")

# Print technology distribution
tech_distribution = {}
for q in cleaned_questions:
    tech = q.get('technology', 'Unknown')
    tech_distribution[tech] = tech_distribution.get(tech, 0) + 1

print("\nTechnology distribution:")
for tech, count in sorted(tech_distribution.items()):
    print(f"{tech}: {count} questions") 