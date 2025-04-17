from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Load the trained model
model = SentenceTransformer("trained-qa-model")

# Load the original questions and answers
file_path = 'combined_interview_questions.json'

with open(file_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

sentence1 = []
sentence2 = []

for item in raw_data:
    q, a = item.get("question", "").strip(), item.get("answer", "").strip()
    if q and a:
        sentence1.append(q)
        sentence2.append(a)

questions = sentence1
answers = sentence2
question_embeddings = model.encode(questions)

# Define the ask_question function
def ask_question(user_input):
    user_embedding = model.encode([user_input])[0]
    similarities = cosine_similarity([user_embedding], question_embeddings)[0]
    max_score = np.max(similarities)
    best_idx = np.argmax(similarities)
    if max_score >= 0.10:
        return f"✅ Answer: {answers[best_idx]} (Score: {max_score:.4f})"
    else:
        return f"❌ Sorry, I don’t have a confident answer. (Highest Score: {max_score:.4f})"

# Ask some questions
user_question_1 = "Explain the concept of inheritance in object-oriented programming."
response_1 = ask_question(user_question_1)
print(response_1)

user_question_2 = "What is the time complexity of a binary search algorithm?"
response_2 = ask_question(user_question_2)
print(response_2)

user_question_3 = "What is nodejs"
response_3 = ask_question(user_question_3)
print(response_3)
