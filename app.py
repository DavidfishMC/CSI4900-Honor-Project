import re
import streamlit as st
from src.detector import load_model

def extract_features(subject, body, has_attachment, att_type, day, hr, device):
    full_text = f"{subject} {body}"
    
    # Text metric calculations
    email_length = len(full_text)
    words = full_text.split()
    word_count = len(words)
    reading_time_minutes = round(word_count / 200, 2)
    
    # Link detection
    urls = re.findall(r'https?://[^\s]+|www\.[^\s]+', full_text)
    num_links = len(urls)
    contains_link = 1 if num_links > 0 else 0
    
    # Attachment mappings (converting boolean checkbox to 1/0)
    att_flag = 1 if has_attachment else 0
    final_att_type = att_type if has_attachment else 'None'
    
    # Character and casing stats
    num_special_characters = len(re.findall(r'[^a-zA-Z0-9\s]', full_text))
    num_uppercase_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    
    # Build the feature dictionary
    sample = {
        'subject': subject,
        'email_body': body,
        'contains_link': contains_link,
        'contains_attachment': att_flag,
        'attachment_type': final_att_type,
        'email_length': email_length,
        'word_count': word_count,
        'reading_time_minutes': reading_time_minutes,
        'num_links': num_links,
        'num_special_characters': num_special_characters,
        'num_uppercase_words': num_uppercase_words,
        'day_of_week': day,
        'hour': hr,
        'device_type': device,
        'email_client': 'Webmail'
    }
    
    return sample

st.set_page_config(page_title="Phishing Email Detector", page_icon="🎣")
st.title("🎣 Phishing Email Detector")
st.caption("Prototype v0: rule-based scoring. Trained ML model integration "
           "(DistilBERT + Random Forest) is in progress.")

SAMPLES = {
    "Fake bank alert": "URGENT: Your account suspended. Verify your account now: "
                       "http://192.168.4.7/login or lose access IMMEDIATELY. "
                       "Confirm your password.",
    "Fake invoice": "Hi, please see the attached invoice.zip and click here "
                    "http://bit.ly/3xYz to confirm your payment details.",
    "Normal email": "Hey, are we still on for lunch tomorrow? "
                    "I'll book the table for noon.",
}

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

subject = st.text_input("Enter Email Subject Line", placeholder="e.g., URGENT: Verify your account")
email_body = st.text_area("Enter Email Content", height=220, placeholder="Paste the body of the email here. Keep all links, do not worry about adding attachements.")
email = f"Subject: {subject}\n\n{email_body}" if subject else email_body

contains_attachment = st.checkbox("Include Attachment")
attachment_type = 'None'
if contains_attachment:
    attachment_type = st.selectbox("Attachment Type", ["pdf", "docx", "zip", "exe"])

col1, col2, col3 = st.columns(3)
with col1:
    day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]) 
with col2:
    hour = st.selectbox("Hour", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]) 
with col3:
    device_type = st.selectbox("Device Type", ["Desktop", "Mobile"])

if st.button("Analyze email", type="primary") and email_body.strip():
    sample_features = extract_features(
        subject, email_body, contains_attachment, attachment_type, 
        day_of_week, hour, device_type
    )
    st.write("Calculated Features Dictionary:", sample_features)

with st.sidebar:
    st.header("Roadmap")
    st.markdown("- Trained ML model (DistilBERT, Sentence-BERT?, Deep Neural Network?, TF-IDF? + Random Forest, XGBoost?, SVM?)\n"
                "- VirusTotal / AbuseIPDB API link checks, .eml` upload (SPF/DKIM/DMARC)?\n"
                "- App features such as Streamlit, wire in trained model into website backend\n"
                "- Security layer like Automated response (SOAR) with Shuffle, ATT&CK Severity Score and Sigma Rules, ansible playbooks?\n"
                "- Elasticsearch logging + Kibana dashboard\n"
                "- Infrastructure such as Dockerized deployment with Docker, Github Actions, Ubuntu VM, Terraform with AWS S3 + IAM\n"
                "- CI: GitHub Actions (pytest, ruff, Bandit, pip-audit, Trivy), CD: Actions builds the image and pushes it to ghcr.io, and Ansible pulls and runs it on the VM\n"
                "- Unlikely Features/In Consideration: FastAPI backend + TypeScript, GoPhish Pen Testing, RAG")
    