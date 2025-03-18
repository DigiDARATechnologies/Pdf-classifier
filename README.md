# PDF Classifier with BERT and Tesseract OCR (Frontend + Backend)

This project classifies PDFs into categories (e.g., `sem_fees`, `exam_timetable`) using BERT for text classification and Tesseract OCR for extracting text from image-based PDFs. It includes a frontend for user interaction, built with Flask.

## Repository

Hosted under the [DigiDara-Technologies](https://github.com/DigiDara-Technologies) organization:  
[https://github.com/DigiDara-Technologies/pdf-classifier](https://github.com/DigiDara-Technologies/pdf-classifier)

## Prerequisites

- **Python 3.8+**
- **Tesseract OCR**:
  - Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
  - Add Tesseract to PATH (e.g., `C:\Program Files\Tesseract-OCR`).
- **Poppler**:
  - Download from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases).
  - Add Poppler to PATH (e.g., `C:\Program Files\poppler-24.02.0\Library\bin`).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DigiDara-Technologies/pdf-classifier.git
   cd pdf-classifier