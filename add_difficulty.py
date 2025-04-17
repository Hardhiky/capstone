import json
import re

def determine_difficulty(question, answer):
    # Convert to lowercase for easier matching
    q_lower = question.lower()
    a_lower = answer.lower()
    
    # Keywords indicating difficulty
    easy_keywords = [
        'what is', 'define', 'explain', 'describe', 'list', 'name',
        'basic', 'simple', 'fundamental', 'difference between',
        'how to', 'syntax', 'example'
    ]
    
    medium_keywords = [
        'how does', 'why', 'when', 'where', 'which', 'compare',
        'implement', 'create', 'write', 'design', 'structure',
        'pattern', 'best practice', 'approach', 'method'
    ]
    
    hard_keywords = [
        'optimize', 'performance', 'scalability', 'architecture',
        'distributed', 'concurrent', 'thread', 'process', 'memory',
        'security', 'encryption', 'algorithm', 'complex', 'advanced',
        'debug', 'troubleshoot', 'error handling', 'exception'
    ]
    
    # Check for code complexity
    code_blocks = len(re.findall(r'```|`', answer))
    if code_blocks > 2:
        return 'Hard'
    
    # Count technical terms
    technical_terms = len(re.findall(r'\b(api|framework|library|protocol|algorithm|pattern|architecture|system|database|server|client|network|security|performance|optimization|scalability|concurrency|thread|process|memory|cache|queue|stack|heap|buffer|stream|socket|proxy|load|balance|cluster|distributed|microservice|container|virtual|cloud|devops|ci|cd|test|debug|deploy|monitor|log|trace|profile|benchmark)\b', q_lower + ' ' + a_lower))
    
    if technical_terms > 5:
        return 'Hard'
    elif technical_terms > 2:
        return 'Medium'
    
    # Check for keywords
    for keyword in hard_keywords:
        if keyword in q_lower or keyword in a_lower:
            return 'Hard'
    
    for keyword in medium_keywords:
        if keyword in q_lower or keyword in a_lower:
            return 'Medium'
    
    for keyword in easy_keywords:
        if keyword in q_lower or keyword in a_lower:
            return 'Easy'
    
    # Default to medium if no clear indicators
    return 'Medium'

# Load the questions
with open('combined_interview_questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

# Add difficulty levels
for q in questions:
    q['difficulty'] = determine_difficulty(q['question'], q['answer'])

# Save the updated dataset
with open('combined_interview_questions.json', 'w', encoding='utf-8') as f:
    json.dump(questions, f, indent=2)

# Print difficulty distribution
difficulty_distribution = {}
tech_difficulty = {}

for q in questions:
    diff = q['difficulty']
    tech = q['technology']
    
    # Overall difficulty distribution
    difficulty_distribution[diff] = difficulty_distribution.get(diff, 0) + 1
    
    # Technology-specific difficulty distribution
    if tech not in tech_difficulty:
        tech_difficulty[tech] = {'Easy': 0, 'Medium': 0, 'Hard': 0}
    tech_difficulty[tech][diff] += 1

print("Overall difficulty distribution:")
for diff, count in sorted(difficulty_distribution.items()):
    print(f"{diff}: {count} questions")

print("\nTechnology-specific difficulty distribution:")
for tech in sorted(tech_difficulty.keys()):
    print(f"\n{tech}:")
    for diff in ['Easy', 'Medium', 'Hard']:
        count = tech_difficulty[tech][diff]
        print(f"  {diff}: {count} questions") 