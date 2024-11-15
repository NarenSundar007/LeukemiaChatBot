import streamlit as st
import easyocr
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from groq import Groq

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """ 
    You are a highly knowledgeable assistant specialized in leukemia and blood cancer. Answer the user's question with as much detail as possible using the provided context. Ensure your response is clear, factual, and medically accurate. 
    If the answer is not in the provided context, respond only if the question pertains to leukemia or blood cancer-related topics. Avoid spelling mistakes, keep the response professional and empathetic, and avoid inappropriate or overly complex language.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:\n"""

    # Use LLaMA 3.1 model via Ollama
    model = Ollama(model="llama3.1")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    print(docs)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    with st.chat_message("assistant"):
        st.write("Reply: ", response["output_text"])



def nutritional_guide():
    st.subheader("Nutritional Guide for Leukemia Patients")
    st.write("Provide the following details to receive a personalized nutritional guide:")

    # Patient details
    age = st.number_input("Age", min_value=0, max_value=120, step=1, help="Enter the patient's age.")
    weight = st.number_input("Weight (kg)", min_value=0, max_value=300, step=1, help="Enter the patient's weight.")
    dietary_restrictions = st.text_area("Dietary Restrictions", help="Mention any dietary restrictions or allergies (e.g., gluten-free, lactose intolerance).")
    preferred_foods = st.text_area("Preferred Foods", help="List foods the patient prefers or enjoys eating.")

    # Specific nutritional needs
    st.write("Nutritional Goals (Check all that apply):")
    improve_energy = st.checkbox("Improve energy levels")
    boost_immunity = st.checkbox("Boost immunity")
    manage_side_effects = st.checkbox("Manage chemotherapy side effects (e.g., nausea, appetite loss)")

    # Additional symptoms or conditions
    st.write("Additional Symptoms or Conditions:")
    anemia = st.checkbox("Anemia")
    weight_loss = st.checkbox("Unintended weight loss")
    fatigue = st.checkbox("Fatigue")

    # Button to generate nutritional guide
    if st.button("Generate Nutritional Guide"):
        st.write("### Personalized Nutritional Guide")

        # Collecting inputs into a single structured summary
        context = f'''
        Patient Details:
        - Age: {age}
        - Weight: {weight} kg
        - Dietary Restrictions: {dietary_restrictions or "None"}
        - Preferred Foods: {preferred_foods or "None"}

        Nutritional Goals:
        - Improve Energy Levels: {"Yes" if improve_energy else "No"}
        - Boost Immunity: {"Yes" if boost_immunity else "No"}
        - Manage Chemotherapy Side Effects: {"Yes" if manage_side_effects else "No"}

        Additional Symptoms or Conditions:
        - Anemia: {"Yes" if anemia else "No"}
        - Unintended Weight Loss: {"Yes" if weight_loss else "No"}
        - Fatigue: {"Yes" if fatigue else "No"}
        '''

        # Define the prompt
        prompt = f'''
        You are a highly knowledgeable assistant specializing in leukemia and blood cancer. Based on the following patient details, generate a personalized nutritional guide:

        {context}
        '''

        try:
            # Initialize the Groq client
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

            # Create a chat completion
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract and display the generated nutritional guide
            nutritional_guide = response.choices[0].message.content
            st.write(nutritional_guide)

        except Exception as e:
            st.error(f"Error generating nutritional guide: {str(e)}")






def save_uploaded_image(uploaded_file, save_dir="uploaded_images"):
    """
    Save the uploaded file to a specified directory on the local system.
    """
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the file to the directory
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    return file_path


def save_uploaded_image(uploaded_file, save_dir="uploaded_images"):
    """
    Save the uploaded file to a specified directory on the local system.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path


def extract_text_from_image_easyocr(image_path):
    """
    Extract text from an image file using EasyOCR.
    """
    reader = easyocr.Reader(['en'], gpu=False)  # Initialize EasyOCR reader
    result = reader.readtext(image_path, detail=0)  # Extract text (detail=0 returns only the text)
    return " ".join(result)


def diagnose_leukemia_groq(report_text):
    """
    Diagnose leukemia using Groq based on the extracted lab report text.
    """
    prompt_template = """ 
    You are a highly knowledgeable assistant specializing in leukemia and blood cancer. Based on the provided lab report text, analyze and suggest whether there is any indication of leukemia. If applicable, suggest the next steps for further diagnosis or treatment. Ensure your response is clear and professional.

    Lab Report Text:
    {report_text}

    Diagnosis and Recommendations:
    """

    prompt = prompt_template.format(report_text=report_text)

    # Initialize Groq client
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # Create a chat completion
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the generated content
    diagnosis = response.choices[0].message.content
    return diagnosis


def diagnosis_assistant():
    """
    Lab Report Diagnosis Assistant using EasyOCR for text extraction and Groq for diagnosis.
    """
    st.header("Lab Report Diagnosis Assistant (Powered by EasyOCR and Groq LLM)")

    # File uploader accepts image files only
    report_file = st.file_uploader("Upload a Lab Report (Image only)", type=["png", "jpg", "jpeg"])
    if report_file:
        st.write("Saving the uploaded image locally...")

        # Save the uploaded image locally
        image_path = save_uploaded_image(report_file)
        st.write(f"Image saved to: `{image_path}`")

        # Extract text from the saved image using EasyOCR
        try:
            lab_report_text = extract_text_from_image_easyocr(image_path)
            st.write("### Extracted Lab Report Text:")
            st.write(lab_report_text)

            if st.button("Diagnose"):
                with st.spinner("Diagnosing based on the lab report..."):
                    diagnosis = diagnose_leukemia_groq(lab_report_text)  # Pass extracted text to Groq LLM
                    st.write("### Diagnosis and Recommendations:")
                    st.write(diagnosis)
        except Exception as e:
            st.error(f"Error processing the image with EasyOCR or Groq: {str(e)}")


def main():
    st.set_page_config("Blood Cancer Query and Diagnosis")

    # Tabs for the two functionalities
    tab1, tab2, tab3 = st.tabs(["General Queries", "Nutrional Assistant", "Diagnosis Assistant"])

    with tab1:
        st.header("Ask your Blood Cancer Related Queries here!")

        with st.chat_message("user"):
            user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

    with tab2:
        nutritional_guide()

    with tab3:
        diagnosis_assistant()



if __name__ == "__main__":
    main()
