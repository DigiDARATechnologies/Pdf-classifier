import os
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pdf2image import convert_from_path
import pytesseract
import joblib

document_categories = {
    "exam_fees": r"C:\Users\ranje\Desktop\Intern Task\Task-4 class manager\project\backend\data\exam_fees",
    "exam_timetable": r"C:\Users\ranje\Desktop\Intern Task\Task-4 class manager\project\backend\data\exam_timetable",
    "exam_result": r"C:\Users\ranje\Desktop\Intern Task\Task-4 class manager\project\backend\data\exam_result",
    "hall_ticket": r"C:\Users\ranje\Desktop\Intern Task\Task-4 class manager\project\backend\data\hall_ticket",
    "sem_fees": r"C:\Users\ranje\Desktop\Intern Task\Task-4 class manager\project\backend\data\sem_fees",
    "scholarship": r"C:\Users\ranje\Desktop\Intern Task\Task-4 class manager\project\backend\data\scholarship",
    "event": r"C:\Users\ranje\Desktop\Intern Task\Task-4 class manager\project\backend\data\event",
    "sports_event": r"C:\Users\ranje\Desktop\Intern Task\Task-4 class manager\project\backend\data\sports",
    "staff_circular": r"C:\Users\ranje\Desktop\Intern Task\Task-4 class manager\project\backend\data\staff",
    "general": r"C:\Users\ranje\Desktop\Intern Task\Task-4 class manager\project\backend\data\genral"
}

# Function to convert PDF to text and save as TXT with manual intervention
def convert_pdf_to_txt(pdf_path):
    txt_path = pdf_path.replace('.pdf', '.txt')
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if pdf_reader.pages:
                text = pdf_reader.pages[0].extract_text() or ""
        print(f"PyPDF2 extracted text from {pdf_path}: {text[:500]}")
    except Exception as e:
        print(f"PyPDF2 failed for {pdf_path}: {e}")
    if not text.strip():
        print(f"Attempting OCR for {pdf_path}")
        try:
            images = convert_from_path(pdf_path, poppler_path=r"C:\Program Files\poppler-24.02.0\Library\bin")
            text = pytesseract.image_to_string(images[0])
            print(f"OCR extracted text from {pdf_path}: {text[:500]}")
        except Exception as e:
            print(f"OCR failed for {pdf_path}: {e} (Tesseract not installed or not in PATH?)")
    if not text.strip():
        print(f"Warning: No text extracted automatically from {pdf_path}")
        user_input = input(f"Manual intervention required for {pdf_path}. Paste the text (or press Enter to skip): ")
        if user_input.strip():
            text = user_input.strip()
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)
            print(f"Manual text saved to {txt_path}")
        else:
            print(f"No manual text provided, no TXT file created")
    elif text.strip():
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
        print(f"Text saved to {txt_path}")
    return text

# Load dataset with automatic TXT conversion
def load_dataset():
    data = []
    labels = []
    label_map = {category: idx for idx, category in enumerate(document_categories.keys())}
    category_counts = {category: 0 for category in document_categories}
    for category, path in document_categories.items():
        if not os.path.exists(path):
            print(f"Directory {path} does not exist. Skipping {category}.")
            continue
        for file in os.listdir(path):
            if file.endswith(".pdf"):
                file_path = os.path.join(path, file)
                text = convert_pdf_to_txt(file_path)
                if text.strip():
                    data.append(text)
                    labels.append(label_map[category])
                    category_counts[category] += 1
                # Use existing TXT file as fallback
                txt_path = file_path.replace('.pdf', '.txt')
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as txt_file:
                        data.append(txt_file.read())
                        labels.append(label_map[category])
                        category_counts[category] += 1
    print(f"Dataset size: {len(data)} PDFs")
    print("Category distribution:", category_counts)
    return data, labels, label_map

# Vector Space Model with TF-IDF
def create_vsm(data):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', vocabulary=['fee', 'tuition', 'semester', 'amount', 'due', 'timetable', 'exam', 'schedule', 'sports', 'event'])
    X = vectorizer.fit_transform(data).toarray()
    return X, vectorizer

# Custom Dataset for PyTorch
class PDFVectorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define a simple ANN
class PDFClassifierANN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PDFClassifierANN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Train the ANN
def train_ann(X_train, y_train, X_test, y_test, num_classes):
    input_size = X_train.shape[1]
    hidden_size = 128
    batch_size = 8
    epochs = 50
    learning_rate = 0.001

    train_dataset = PDFVectorDataset(X_train, y_train)
    test_dataset = PDFVectorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = PDFClassifierANN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")

    return model, accuracy

# Classify a new PDF with TXT conversion
def classify_pdf(pdf_path, vectorizer, model, label_map):
    text = convert_pdf_to_txt(pdf_path)
    print(f"Extracted text from {pdf_path}: {text[:500]}")
    if not text.strip():
        print(f"No text extracted, defaulting to 'general'")
        return "general"
    X_new = vectorizer.transform([text]).toarray()
    X_new = torch.tensor(X_new, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(X_new)
        print(f"Logits: {outputs}")
        _, predicted = torch.max(outputs.data, 1)
    index_to_category = {v: k for k, v in label_map.items()}  # Reverse mapping
    predicted_category = index_to_category[predicted.item()]
    print(f"Predicted index: {predicted.item()}, Category: {predicted_category}")
    return predicted_category

if __name__ == "__main__":
    # Ensure the models directory exists
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'project', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    data, labels, label_map = load_dataset()
    print(f"Dataset size: {len(data)} PDFs")
    print("Category distribution:", {k: data.count(k) for k in label_map.keys() if k in data})
    X, vectorizer = create_vsm(data)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    num_classes = len(label_map)
    model, final_accuracy = train_ann(X_train, y_train, X_test, y_test, num_classes)
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
    # Save the model and dependencies with full paths
    torch.save(model.state_dict(), os.path.join(model_dir, 'ann_model.pth'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    joblib.dump(label_map, os.path.join(model_dir, 'label_map.pkl'))

    # Load saved models with full paths
    label_map = joblib.load(os.path.join(model_dir, 'label_map.pkl'))
    vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    model.load_state_dict(torch.load(os.path.join(model_dir, 'ann_model.pth')))
    model.eval()

    # Test with a different PDF
    new_pdf_path = r"C:\Users\ranje\Downloads\B.E-B.Tech-V-Sem-Tution-Fee-26.07.2024.pdf"
    category = classify_pdf(new_pdf_path, vectorizer, model, label_map)
    print(f"Predicted Category for new PDF: {category}")