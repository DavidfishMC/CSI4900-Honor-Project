import streamlit as st
from src.detector import load_model, analyze

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

choice = st.selectbox("Load a sample email", ["(paste your own)"] + list(SAMPLES))
default = SAMPLES.get(choice, "")
email = st.text_area("Email content", value=default, height=220,
                     key=f"email_{choice}")

if st.button("Analyze email", type="primary") and email.strip():
    r = analyze(email)

    if r["label"] == "phishing":
        st.error("⚠️ Likely phishing")
    else:
        st.success("✅ Looks safe")

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk score", f"{r['score']}/100")
    c2.metric("Severity", r["severity"])
    c3.metric("MITRE ATT&CK", r["attack_tag"].split(" ")[0])
    st.caption(f"ATT&CK technique: {r['attack_tag']}")

    st.subheader("Why it was flagged")
    if r["signals"]:
        st.table({"Signal": list(r["signals"].keys()),
                  "Detail": [str(v) for v in r["signals"].values()]})
    else:
        st.write("No suspicious signals found.")

    if r["urls"]:
        st.subheader("Links found")
        for u in r["urls"]:
            st.code(u)

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
    