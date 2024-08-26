import csv
from transformers import pipeline
from difflib import SequenceMatcher
import pdfplumber

def filter_similar_questions(questions, similarity_threshold=0.7):
    unique_questions = []
    for question in questions:
        is_similar = False
        for uq in unique_questions:
            similarity = SequenceMatcher(None, question['generated_text'], uq['generated_text']).ratio()
            if similarity > similarity_threshold:
                is_similar = True
                break
        if not is_similar:
            unique_questions.append(question)
    return unique_questions

def generate_questions_and_answers(content, chunk_size=512, num_questions=5, temperature=1.0, top_p=0.9):
    # Load pre-trained models for question generation and answering
    question_generator = pipeline('text2text-generation', model='valhalla/t5-base-qg-hl')
    question_answerer = pipeline('question-answering', model='huggingface-course/bert-finetuned-squad')

    # Split content into chunks if it exceeds the chunk_size
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

    results = []
    for chunk in chunks:
        # Generate multiple questions for each chunk
        generated = question_generator(
            f'generate questions: {chunk}', 
            max_length=chunk_size, 
            num_beams=num_questions * 2, # Increase num_beams to match or exceed num_return_sequences
            num_return_sequences=num_questions * 2,  # Generate more questions to filter
            temperature=temperature,
            top_p=top_p,
            early_stopping=True
        )

        # Filter out similar questions to ensure variety
        filtered_questions = filter_similar_questions(generated, similarity_threshold=0.7)

        # Limit to the desired number of questions
        filtered_questions = filtered_questions[:num_questions]

        for question in filtered_questions:
            q = question['generated_text']
            # Answer each question based on the content
            answer = question_answerer(question=q, context=chunk)
            complete_answer = answer['answer'].rstrip()
            
            # Ensure the answer is complete and ends with a period
            if not complete_answer.endswith('.'):
                if '.' in complete_answer:
                    complete_answer = complete_answer[:complete_answer.rfind('.')+1]
                else:
                    # If there's no period, we assume the answer is complete if it's long enough
                    complete_answer = complete_answer

            results.append((q, complete_answer))

    return results

def save_to_csv(questions_and_answers, filename='questions_answers.csv'):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Content'])
        for q, a in questions_and_answers:
            writer.writerow([q, a])

def extract_text_from_pdf(file_path):
    """
    Extracts and returns the text content from a given PDF file.

    Args:
    - file_path (str): Path to the PDF file.

    Returns:
    - str: The content of the PDF file as a string.
    """
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return None


def extract_text_from_file(file_path):
    """
    Extracts and returns the text content from a given text file.

    Args:
    - file_path (str): Path to the text file.

    Returns:
    - str: The content of the file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


title = "Major types of soil in India"
content = extract_text_from_file(r"C:\VSCode\isro-Website - Copy\Add data\soil.txt")

questions_and_answers = generate_questions_and_answers(content, chunk_size=1024, num_questions=5)
save_to_csv(questions_and_answers, 'questions_answers.csv')

print("Questions and answers have been saved to 'questions_answers.csv'")
