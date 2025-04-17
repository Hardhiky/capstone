from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import json
import time

# Set up headless Chrome
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# Start WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# InterviewBit Python Interview Questions URL
URL = "https://www.interviewbit.com/python-interview-questions/"

# Open the page
driver.get(URL)

# Wait for JavaScript content to load
time.sleep(5)

# Sections to scrape
SECTIONS = {
    "Freshers": "freshers",
    "Experienced": "experienced",
    "Python OOPs": "python-oops"
}

questions = []

for category, section_id in SECTIONS.items():
    try:
        # Find section header
        section_element = driver.find_element(By.ID, section_id)

        # Find all question elements (h3 or strong tags)
        question_elements = section_element.find_elements(By.XPATH, ".//h3 | .//strong")

        for question_element in question_elements:
            question_text = question_element.text.strip()

            # Find the next sibling paragraph (p) for the answer
            try:
                answer_element = question_element.find_element(By.XPATH, "./following-sibling::*[1]")
                expected_answer = answer_element.text.strip()
            except:
                expected_answer = "Answer not available"

            # Add to dataset
            questions.append({
                "question": question_text,
                "category": category,
                "expected_answer": expected_answer,
                "follow_up": "Can you provide an optimized approach?",
                "tags": ["python", category.lower().replace(" ", "-")]
            })

    except Exception as e:
        print(f"❌ Error scraping {category}: {e}")

# Save to JSON
with open("interviewbit_python_questions.json", "w") as f:
    json.dump(questions, f, indent=4)

# Close browser
driver.quit()

print(f"✅ Scraped {len(questions)} Python interview questions from InterviewBit!")

