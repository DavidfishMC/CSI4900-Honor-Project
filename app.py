import re
import streamlit as st
from src.detector import load_model
import joblib

import base64
import requests

@st.cache_resource
def load_artifacts():
    model = joblib.load('phishing_ensemble_model.pkl')
    selector = joblib.load('selector.pkl')
    selector_mi = joblib.load('selector_mi.pkl')
    all_cols = pd.read_csv("x_columns_only.csv", nrows=0).columns
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    b_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    b_model.eval()
    
    return model, selector, selector_mi, all_cols, tokenizer, b_model, device

model, selector, selector_mi, all_cols, tokenizer, b_model, device = load_artifacts()

def classify_email(subject, email_body, contains_attachment, attachment_type, day_of_week, hour, device_type, email_client):
    # Calculate text metrics dynamically
    email_length = len(subject) + len(email_body)
    word_count = len((subject + " " + email_body).split())
    reading_time_minutes = word_count / 200.0
    num_links = len(re.findall(r'https?://[^\s]+|www\.[^\s]+', email_body))
    num_special_characters = sum(1 for c in (subject + email_body) if not c.isalnum() and not c.isspace())
    num_uppercase_words = sum(1 for w in (subject + email_body).split() if w.isupper())
    contains_link = 1 if num_links > 0 else 0

    sample = {
        'subject': subject, 'email_body': email_body,
        'contains_link': contains_link, 'contains_attachment': 1 if contains_attachment else 0,
        'attachment_type': attachment_type, 'email_length': email_length,
        'word_count': word_count, 'reading_time_minutes': reading_time_minutes,
        'num_links': num_links, 'num_special_characters': num_special_characters,
        'num_uppercase_words': num_uppercase_words, 'day_of_week': day_of_week,
        'hour': hour, 'device_type': device_type, 'email_client': email_client
    }

    df_s = pd.get_dummies(pd.DataFrame([sample]), drop_first=True)
    tabular_cols = [c for c in all_cols if not c.startswith("bert_")]
    df_s = df_s.reindex(columns=tabular_cols, fill_value=0)

    text = [(subject + " " + email_body)]
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        b_emb = b_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()

    b_df = pd.DataFrame(b_emb, columns=[f"bert_{i}" for i in range(768)])

    X_raw_combined = pd.concat([df_s.reset_index(drop=True), b_df], axis=1)
    X_raw_combined = X_raw_combined.reindex(columns=all_cols, fill_value=0)

    X_var_filtered = selector.transform(X_raw_combined) 
    X_sample = selector_mi.transform(X_var_filtered)         

    probs = model.predict_proba(X_sample)[0]
    return probs[1] # Phishing probability

def check_virustotal_links(urls):
    API_KEY = st.secrets["VIRUSTOTAL_API_KEY"]
    headers = {"x-apikey": API_KEY}
    
    for url in urls:
        url_id = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
        endpoint = f"https://www.virustotal.com/api/v3/urls/{url_id}"
        
        try:
            response = requests.get(endpoint, headers=headers, timeout=5)
            if response.status_code == 200:
                stats = response.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
                if stats.get("malicious", 0) > 0 or stats.get("suspicious", 0) > 0:
                    return True
        except Exception:
            pass # Fail silently or handle gracefully without UI popups
            
    return False

def extract_features(subject, body, has_attachment, att_type, day, hr, device, client):
    full_text = f"{subject} {body}".strip() if subject else body.strip()
    
    # Text metric calculations
    email_length = len(body)
    
    # Count words based exclusively on the email body content
    body_words = body.split() if body else []
    word_count = len(body_words)
    
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
    num_uppercase_words = sum(1 for w in body_words if w.isupper() and len(w) > 1)
    
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
        'email_client': client
    }
    
    return sample

st.set_page_config(page_title="Phishing Email Detector", page_icon="🎣")
st.title("🎣 Phishing Email Detector")
st.caption("Think your email might be phishing? Input the email's information in the provided fields to find out!")

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
def load_model():
    return joblib.load('phishing_ensemble_model.pkl')

model = load_model()

subject = st.text_input("Enter Email Subject Line", placeholder="e.g., URGENT: Verify your account")
email_body = st.text_area("Enter Email Content", height=220, placeholder="Paste the body of the email here. Keep all links, do not worry about adding attachements.")
email = f"Subject: {subject}\n\n{email_body}" if subject else email_body

contains_attachment = st.checkbox("Select if your email contains a File Attachment")
attachment_type = 'None'
if contains_attachment:
    attachment_type = st.selectbox("Provide the file type of one of the Attachment(s)", ["pdf", "docx", "zip", "exe"])

hour_options = {
    "12:00 AM (Midnight)": 0, "1:00 AM": 1, "2:00 AM": 2, "3:00 AM": 3,
    "4:00 AM": 4, "5:00 AM": 5, "6:00 AM": 6, "7:00 AM": 7,
    "8:00 AM": 8, "9:00 AM": 9, "10:00 AM": 10, "11:00 AM": 11,
    "12:00 PM (Noon)": 12, "1:00 PM": 13, "2:00 PM": 14, "3:00 PM": 15,
    "4:00 PM": 16, "5:00 PM": 17, "6:00 PM": 18, "7:00 PM": 19,
    "8:00 PM": 20, "9:00 PM": 21, "10:00 PM": 22, "11:00 PM": 23
}

col1, col2, col3, col4 = st.columns(4)
with col1:
    day_of_week = st.selectbox("Day of Week Sent", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]) 
with col2:
    selected_hour_label = st.selectbox("Hour Sent", list(hour_options.keys()))
    hour = hour_options[selected_hour_label]
with col3:
    device_type = st.selectbox("Device Received On", ["Laptop", "Desktop", "Mobile", "Tablet"])
with col4:
    email_client = st.selectbox("Email Client Received On", ["Thunderbird", "Gmail", "Apple Mail", "Yahoo Mail", "Outlook"])

btn_col, msg_col = st.columns([1, 2])

with btn_col:
    analyze_clicked = st.button("Analyze email", type="primary")

with msg_col:
    if analyze_clicked:
        if not subject.strip() and not email_body.strip():
            st.error("⚠️ Please enter a subject line and email content, and double-check your other inputs!")
        elif not subject.strip():
            st.error("⚠️ Please enter a subject line, and double-check your other inputs!")
        elif not email_body.strip():
            st.error("⚠️ Please enter email content, and double-check your other inputs!")

if analyze_clicked and subject.strip() and email_body.strip():
    # Extract links from the combined text first
    combined_text = f"{subject} {email_body}"
    extracted_urls = re.findall(r'https?://[^\s]+|www\.[^\s]+', combined_text)
    
    # Check links against VirusTotal before running feature extraction or ML prediction
    vt_flagged = check_virustotal_links(extracted_urls) if extracted_urls else False
    
    if vt_flagged:
        st.error("**Threat Detected:** VirusTotal flagged one or more links in this email as malicious. This email is classified as **Phishing** with a **100%** probability.")
    else:
        phishing_prob = classify_email(
            subject, email_body, contains_attachment, attachment_type, 
            day_of_week, hour, device_type, email_client
        )
        
        st.subheader("Analysis Results")
        st.metric(label="Phishing Probability", value=f"{phishing_prob * 100:.2f}%")
        
        if phishing_prob >= 0.5:
            st.error(f"**Result:** High risk detected! This email is classified as **Phishing** with a **{phishing_prob * 100:.2f}%** probability.")
        else:
            st.success(f"**Result:** This email appears to be **Legitimate** (Phishing Probability: **{phishing_prob * 100:.2f}%**).")

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
    