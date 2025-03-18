import joblib
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a given PDF file.
    """
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text

def classify_pdf(pdf_path):
    """
    Predicts the category of the uploaded PDF.
    """
    # Load the pre-trained model and vectorizer
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)

    # Transform the text using the vectorizer
    text_vector = vectorizer.transform([text])

    # Predict the category using the model
    prediction = model.predict(text_vector)
    return prediction[0]