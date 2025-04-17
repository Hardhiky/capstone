# Step 1: Install required packages with compatible versions
# !pip install sentence-transformers huggingface-hub scikit-learn datasets transformers  # Run this in your local terminal

import json
import torch
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import os

# Download all required NLTK data with error handling
def download_nltk_data():
    required_nltk_data = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
    for item in required_nltk_data:
        try:
            nltk.download(item, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {item}: {str(e)}")
            print("Continuing with available resources...")

# Download NLTK data
download_nltk_data()

# Step 4: Verify installed versions (optional)
import huggingface_hub
import transformers
import sentence_transformers

print(f"huggingface-hub version: {huggingface_hub.__version__}")
print(f"transformers version: {transformers.__version__}")
print(f"sentence-transformers version: {sentence_transformers.__version__}")

# Step 5: Load JSON file
file_path = 'combined_interview_questions.json'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Make sure it's in the same directory as this script.")
    exit()

# Step 6: Extract Q&A data and preprocess
sentence1 = []
sentence2 = []
difficulties = []

def preprocess_text(text):
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Simple word tokenization without NLTK if it fails
        try:
            words = word_tokenize(text)
        except:
            words = text.split()
        
        # Remove stopwords if available
        try:
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words]
        except:
            # If stopwords are not available, just use the words as is
            pass
        
        return ' '.join(words)
    except Exception as e:
        print(f"Warning: Error in preprocessing text: {str(e)}")
        return text  # Return original text if preprocessing fails

for item in raw_data:
    q = item.get("question", "").strip()
    a = item.get("answer", "").strip()
    diff = item.get("difficulty", "Medium")
    
    if q and a:
        # Preprocess question and answer
        q_processed = preprocess_text(q)
        a_processed = preprocess_text(a)
        
        # Add original and processed versions
        sentence1.extend([q, q_processed])
        sentence2.extend([a, a_processed])
        difficulties.extend([diff, diff])

# Step 7: Create Hugging Face dataset
hf_dataset = Dataset.from_dict({
    "sentence1": sentence1,
    "sentence2": sentence2,
    "difficulty": difficulties
})

# Step 8: Load pretrained model and define loss
# Use a more advanced model
model = SentenceTransformer('all-mpnet-base-v2')
train_loss = losses.MultipleNegativesRankingLoss(model)

# Add data augmentation for better generalization
def augment_text(text):
    words = word_tokenize(text)
    augmented_texts = []
    
    if len(words) > 3:
        # Randomly remove some words
        for _ in range(2):
            words_copy = words.copy()
            remove_idx = np.random.randint(0, len(words_copy))
            words_copy.pop(remove_idx)
            augmented_texts.append(' '.join(words_copy))
        
        # Randomly replace some words with synonyms
        from nltk.corpus import wordnet
        for _ in range(2):
            words_copy = words.copy()
            replace_idx = np.random.randint(0, len(words_copy))
            word = words_copy[replace_idx]
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
            if synonyms:
                words_copy[replace_idx] = np.random.choice(synonyms)
                augmented_texts.append(' '.join(words_copy))
    
    return augmented_texts

# Augment the training data
augmented_sentence1 = []
augmented_sentence2 = []
augmented_difficulties = []

for s1, s2, diff in zip(sentence1, sentence2, difficulties):
    augmented_texts = augment_text(s1)
    for aug_text in augmented_texts:
        augmented_sentence1.append(aug_text)
        augmented_sentence2.append(s2)
        augmented_difficulties.append(diff)

# Combine original and augmented data
sentence1.extend(augmented_sentence1)
sentence2.extend(augmented_sentence2)
difficulties.extend(augmented_difficulties)

# Step 9: Convert to InputExamples
train_samples = [{"texts": [s1, s2]} for s1, s2 in zip(sentence1, sentence2)]

# Shuffle the data
np.random.shuffle(train_samples)

train_examples = [
    InputExample(texts=example["texts"])
    for example in train_samples
]

# Create evaluation dataset (using 10% of the data)
eval_size = max(int(len(train_examples) * 0.1), 100)  # Ensure at least 100 examples
train_examples_final = train_examples[:-eval_size]
eval_examples = train_examples[-eval_size:]

# Create train dataloader with the training examples
train_dataloader = DataLoader(train_examples_final, shuffle=True, batch_size=4)  # Reduced batch size for local training

# Create evaluator with proper similarity scores
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import torch

class CustomEmbeddingSimilarityEvaluator(EmbeddingSimilarityEvaluator):
    def __init__(self, sentences1, sentences2, scores, name=''):
        super().__init__(sentences1, sentences2, scores, name=name)
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
    
    @classmethod
    def from_input_examples(cls, examples, name=''):
        sentences1 = [example.texts[0] for example in examples]
        sentences2 = [example.texts[1] for example in examples]
        # Create varying scores based on text similarity
        scores = []
        for s1, s2 in zip(sentences1, sentences2):
            # Simple text similarity score
            words1 = set(s1.lower().split())
            words2 = set(s2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            if union:
                score = len(intersection) / len(union)
            else:
                score = 0.0
            scores.append(score)
        return cls(sentences1, sentences2, scores, name=name)
    
    def compute_metrics(self, model):
        # Compute embeddings
        embeddings1 = model.encode(self.sentences1, convert_to_tensor=True)
        embeddings2 = model.encode(self.sentences2, convert_to_tensor=True)
        
        # Compute cosine similarities
        similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        
        # Convert to numpy for correlation computation
        similarities = similarities.cpu().numpy()
        scores = np.array(self.scores)
        
        # Add small noise to prevent constant inputs
        noise = np.random.normal(0, 0.01, size=scores.shape)
        scores = scores + noise
        
        # Compute correlations with error handling
        from scipy.stats import pearsonr, spearmanr
        try:
            eval_pearson, _ = pearsonr(scores, similarities)
            eval_spearman, _ = spearmanr(scores, similarities)
        except:
            # If correlation fails, use mean similarity as metric
            eval_pearson = np.mean(similarities)
            eval_spearman = np.mean(similarities)
        
        return {
            'eval_similarity_pearson_cosine': eval_pearson,
            'eval_similarity_spearman_cosine': eval_spearman,
            'eval_mean_similarity': float(np.mean(similarities))
        }

# Create evaluator
evaluator = CustomEmbeddingSimilarityEvaluator.from_input_examples(
    eval_examples,
    name='eval_similarity'
)

# Step 10: Move model to appropriate device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA. Device name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU.")
model.to(device)

# Step 11: Train the model with improved parameters
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=30,
    warmup_steps=50,  # Reduced warmup steps for smaller batch size
    show_progress_bar=True,
    evaluation_steps=50,  # Reduced evaluation steps for faster feedback
    save_best_model=True
)

# Step 12: Save the trained model
model.save("trained-qa-model")
print("✅ Model training complete and saved.")

# Step 13: Set up for inference
questions = sentence1
answers = sentence2
question_embeddings = model.encode(questions)

# Step 14: Define an improved function to ask questions
def ask_question(user_input):
    # Preprocess the user input
    processed_input = preprocess_text(user_input)
    
    # Get embeddings for both original and processed input
    user_embedding = model.encode([user_input])[0]
    processed_embedding = model.encode([processed_input])[0]
    
    # Calculate similarities for both embeddings
    similarities1 = cosine_similarity([user_embedding], question_embeddings)[0]
    similarities2 = cosine_similarity([processed_embedding], question_embeddings)[0]
    
    # Take the maximum similarity between original and processed input
    similarities = np.maximum(similarities1, similarities2)
    
    # Get top 5 most similar questions
    top_k = 5
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_scores = similarities[top_indices]
    
    # If the best match is very confident, return just that
    if top_scores[0] >= 0.85:
        return f"✅ Answer: {answers[top_indices[0]]} (Score: {top_scores[0]:.4f})"
    
    # Otherwise, return a combined response from top matches
    response = "Here are the most relevant answers I found:\n\n"
    for idx, score in zip(top_indices, top_scores):
        if score >= 0.6:  # Only include reasonably good matches
            response += f"• {answers[idx]} (Score: {score:.4f})\n"
    
    if response == "Here are the most relevant answers I found:\n\n":
        return "❌ I couldn't find a confident answer to your question. Please try rephrasing or ask a different question."
    
    return response

# Step 15: Test with an example question
print(ask_question("What is a list comprehension in python?"))

# Step 16: Evaluate accuracy on the training set
correct = 0
for i, question in enumerate(questions):
    result = ask_question(question)
    if answers[i].lower() in result.lower():
        correct += 1

accuracy = correct / len(questions) * 100
print(f"📊 Training Set Accuracy: {accuracy:.2f}%")
