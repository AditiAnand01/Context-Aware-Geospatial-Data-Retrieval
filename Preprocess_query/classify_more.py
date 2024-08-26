import re

def extract_tag(question, tags):
    """
    Extracts the first matching tag from the question.
    
    Parameters:
    question (str): The question from which to extract the tag.
    tags (list): A list of tags to search for in the question.
    
    Returns:
    str: The first tag found in the question or 'find' if no tag is found.
    """
    question_lower = question.lower()  # Convert question to lowercase for case-insensitive search
    for tag in tags:
        if tag in question_lower:
            return tag
    return 'find'  # Return 'find' if no tag is present

def process_questions(questions):
    """
    Processes a list of questions to extract tags.
    
    Parameters:
    questions (list): A list of questions (strings).
    
    Returns:
    list: A list of tuples, each containing the question and the extracted tag.
    """
    # Define the tags you want to search for
    tags = ['what', 'which', 'how', 'where', 'when', 'why', 'is', 'are', 'find', 'could', 
            'do', 'should', 'would', 'can', 'will', 'who', 'whom', 'does', 'shall', 'may', 
            'might', 'has', 'have', 'had']
    
    # Process each question and extract the tag
    tagged_questions = [(q, extract_tag(q, tags)) for q in questions]
    
    return tagged_questions

## Example usage:
# questions = [
#     "What are the famous places in India?",
#     "Which is the largest city in Australia?",
#     "How can I learn Python programming?",
#     "Where is the Eiffel Tower located?",
#     "Tell me about the weather tomorrow.",
#     "Which flood had occured."
# ]

# # Get the tagged questions
# tagged_questions = process_questions(questions)

# # Print the results
# for question, tag in tagged_questions:
#     print(f"Question: {question}")
#     print(f"Tag: {tag}\n")
