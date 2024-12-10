# **Blood Cancer Query and Diagnosis Assistant**

An AI-powered Streamlit application designed to assist patients and healthcare professionals with leukemia and blood cancer-related queries. This app integrates functionalities for general question answering, nutritional guidance for leukemia patients, and lab report diagnosis.

---

## **Table of Contents**
1. [Features](#features)
2. [Technology Stack](#technology-stack)
3. [Installation](#installation)
4. [Usage](#usage)

---

## **Features**
### **1. General Queries**
- Upload PDF files containing leukemia-related information.
- Ask questions related to blood cancer or leukemia based on the uploaded files.
- Get AI-powered, context-specific responses using advanced LLMs.

### **2. Nutritional Assistant**
- Generate personalized nutritional guides for leukemia patients.
- Input details like age, weight, dietary restrictions, and nutritional goals.
- AI-generated guides help improve energy, boost immunity, and manage chemotherapy side effects.

### **3. Lab Report Diagnosis**
- Upload lab report images in formats like PNG, JPG, or JPEG.
- Extract text using **EasyOCR** and diagnose leukemia using an LLM.
- Receive detailed recommendations and next steps for diagnosis or treatment.

---

## **Technology Stack**
- **Frontend:** Streamlit – for building the interactive user interface.
- **Backend:**
  - **Python:** For processing and application logic.
  - **PyPDF2:** For extracting text from PDF files.
  - **EasyOCR:** For text extraction from lab report images.
- **Vector Database:** FAISS – for similarity search.
- **AI Integrations:**
  - **Google Generative AI (Vertex AI):** For embeddings and LLM-powered responses.
  - **Groq API:** For advanced leukemia-related insights.
  - **LangChain:** For conversational AI and chaining workflows.
- **LLMs:** LLaMA and Ollama.

---

## **Installation**
### **Prerequisites**
- Python 3.8 or higher
- Streamlit (`pip install streamlit`)
- EasyOCR (`pip install easyocr`)
- PyPDF2 (`pip install PyPDF2`)
- LangChain and FAISS (`pip install langchain langchain-community faiss-cpu`)
- Google Generative AI (`pip install google-generativeai langchain-google-genai`)
- Groq SDK (`pip install groq`)


