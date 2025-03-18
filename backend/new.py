import os
import PyPDF2
from transformers import pipeline

# PDF Categories
document_categories = {
    "exam_fees": "./data/exam_fees",
    "exam_timetable": "./data/exam_timetable",
    "exam_result": "./data/exam_result",
    "hall_ticket": "./data/hall_ticket",
    "sem_fees": "./data/sem_fees",
    "scholarship": "./data/scholarship",
    "event": "./data/event",
    "sports_event": "./data/sports",
    "staff_circular": "./data/staff",
    "general": "./data/general"
}

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from all pages of a given PDF file.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
            text = text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    if not text:
        print(f"Warning: No text extracted from {pdf_path}")
    else:
        print(f"Extracted Text (first 500 chars): {text[:500]}")
    return text

def classify_pdf(pdf_path):
    """
    Predicts the category of the uploaded PDF using Hugging Face transformers.
    Responds only with the category name.
    """
    text = extract_text_from_pdf(pdf_path)
    categories_list = list(document_categories.keys())
    
    if not text:
        print("No text available to classify, defaulting to 'general'")
        return "general"

    # Zero-shot classification with Hugging Face pipeline
    classifier = pipeline("zero-shot-classification", model="distilbert-base-uncased")
    result = classifier(text, candidate_labels=categories_list)

    predicted_category = result["labels"][0]
    return predicted_category

if __name__ == "__main__":
    pdf_path = r"C:\Users\ranje\Downloads\SPORTS_BOARD_(CIRCULAR)File[1] (1) (1).pdf"
    category = classify_pdf(pdf_path)
    print(f"{category}")