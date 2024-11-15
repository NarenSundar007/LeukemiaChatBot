import os
import json
import PyPDF2
from gensim.summarization import summarize

# Define the folder containing PDFs
pdf_folder = "PDF"

# Output file paths
faqs_output_file = "faqs.json"
summaries_output_file = "summaries.json"

# Initialize data structures
faqs = []
summaries = {}

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def process_faqs(text):
    """Extract FAQs from the text using basic patterns."""
    faq_sections = [section for section in text.split("\n\n") if "?" in section]
    for section in faq_sections:
        question, *answer = section.split("\n")
        if question.endswith("?") and answer:
            faqs.append({
                "question": question.strip(),
                "answer": " ".join(answer).strip()
            })

def summarize_text(text):
    """Summarize long text using gensim summarization."""
    try:
        # Summarize long text
        summary = summarize(text, word_count=150)  # Adjust word count as needed
        return summary if summary.strip() else text[:500]  # Fallback to the first 500 chars
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return text[:500]  # Fallback to the first 500 chars

# Process all PDFs in the folder
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"Processing {filename}...")
        text = extract_text_from_pdf(pdf_path)
        
        # Process FAQs
        process_faqs(text)
        
        # Summarize Guidelines
        summaries[filename] = summarize_text(text)

# Save FAQs to JSON
with open(faqs_output_file, "w", encoding="utf-8") as faqs_file:
    json.dump(faqs, faqs_file, indent=4, ensure_ascii=False)
print(f"FAQs saved to {faqs_output_file}")

# Save Summaries to JSON
with open(summaries_output_file, "w", encoding="utf-8") as summaries_file:
    json.dump(summaries, summaries_file, indent=4, ensure_ascii=False)
print(f"Summaries saved to {summaries_output_file}")
