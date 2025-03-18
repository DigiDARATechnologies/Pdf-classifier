# Class Manager - PDF Classifier

## Overview
This project is a **Class Manager** tool designed to [describe purpose, "organize and classify PDF documents based on their content"]. It aims to streamline [specific goal "document management for educational or professional use"] by leveraging [key technologies, "Python, machine learning, or file parsing techniques"].

## Features
- **PDF Classification:** ["Automatically categorizes PDFs into predefined classes."]
- **User Interface:** ["Simple command-line interface for managing files."]
- **Customizable:** ["Allows users to define classification rules."]

## Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:DigiDARATechnologies/Pdf-classifier.git
   cd Pdf-classifier
   ```
2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```sh
   cd backend
   python pdf_classifier.py
   ```
   Open your browser and go to http://127.0.0.1:5000 to access the frontend.
   The app will:
   - Allow users to upload PDFs via the frontend.
   - Extract text from PDFs using Tesseract OCR.
   - Classify the PDFs using a BERT model.
   - Display the predicted category (e.g., sem_fees).
     
## Directory Structure

- backend/: Contains the backend code.
- - pdf_classifier.py: Main script with BERT, Tesseract, and Flask.
  - static/: CSS/JS files for the frontend.
  - templates/: HTML templates for the frontend.
  - upload/: Directory for uploaded PDFs and model files (not included in repo).
  - requirements.txt: List of dependencies.
  - README.md: Project documentation.

## Notes
 - Ensure Tesseract and Poppler are in your PATH.
 - If OCR fails, check PDF quality or adjust Tesseract settings (--psm 6).
 - Model files (e.g., bert_model.pth) are excluded due to size. Retrain the model or upload via Git LFS.

## License
**MIT License**
