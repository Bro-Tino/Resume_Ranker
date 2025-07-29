import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from operator import itemgetter

from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

# Load Azure credentials
load_dotenv()
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    return text.strip()

# LangChain LLM setup
def get_llm():
    return AzureChatOpenAI(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0
    )

def score_resume(llm, job_desc, resume_text):
    prompt = ChatPromptTemplate.from_template("""
You are an expert hiring assistant. Given a job description and a resume, score how well the resume matches the job out of 100.
Also provide a 1-sentence justification for the score.

Job Description:
{jd}

Resume:
{resume}

Respond in format:
Score: <score>
Reason: <reason>
""")
    chain = prompt | llm
    response = chain.invoke({"jd": job_desc, "resume": resume_text})
    return response.content

def parse_score_response(response):
    lines = response.split("\n")
    score_line = next((l for l in lines if l.lower().startswith("score:")), "Score: 0")
    reason_line = next((l for l in lines if l.lower().startswith("reason:")), "Reason: Not provided.")
    try:
        score = int(score_line.split(":")[1].strip())
    except:
        score = 0
    return score, reason_line.split(":", 1)[1].strip()

# Streamlit UI
st.title("🔍 Resume Ranker with Azure OpenAI")
st.markdown("Upload a job description and up to 10 resumes. The app will score and rank them.")

job_file = st.file_uploader("📄 Upload Job Description (PDF)", type="pdf")

resume_files = st.file_uploader("📁 Upload Resumes (PDFs)", type="pdf", accept_multiple_files=True)

if job_file and resume_files:
    with st.spinner("Reading and analyzing..."):
        job_desc = extract_pdf_text(job_file)
        llm = get_llm()

        results = []
        for resume in resume_files:
            resume_text = extract_pdf_text(resume)
            response = score_resume(llm, job_desc, resume_text)
            score, reason = parse_score_response(response)
            results.append({
                "name": resume.name,
                "score": score,
                "reason": reason
            })

        sorted_results = sorted(results, key=itemgetter("score"), reverse=True)

    st.success("Done! Here are the ranked resumes:")

    for res in sorted_results:
        st.markdown(f"### 🧑 {res['name']}")
        st.write(f"*Score:* {res['score']}/100")
        st.write(f"*Reason:* {res['reason']}")
        st.markdown("---")