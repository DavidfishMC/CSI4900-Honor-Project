import re
import streamlit as st
import pandas as pd
from src.detector import load_model
import joblib
import ijson
import base64
import requests
import torch
import os
import glob
from transformers import DistilBertModel, DistilBertTokenizer
from difflib import SequenceMatcher
from transformers import AutoModel, AutoTokenizer
import tempfile
from email import policy
from email.parser import BytesParser

APP_DIR = os.path.dirname(os.path.abspath(__file__))

def assemble_weights(model_dir: str = APP_DIR, out_filename: str = "model.safetensors") -> str:
    out_path = os.path.join(model_dir, out_filename)
    part_paths = sorted(
        glob.glob(os.path.join(model_dir, f"{out_filename}.part*")),
        key=lambda p: int(p.rsplit(".part", 1)[-1])
    )

    if part_paths:
        expected_size = sum(os.path.getsize(p) for p in part_paths)
        if not os.path.exists(out_path) or os.path.getsize(out_path) != expected_size:
            temp_path = out_path + ".tmp"
            with open(temp_path, "wb") as out:
                for p in part_paths:
                    with open(p, "rb") as inp:
                        while True:
                            buf = inp.read(1024 * 1024 * 8)
                            if not buf:
                                break
                            out.write(buf)
            os.replace(temp_path, out_path)

    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Missing {out_path} and no split part files found in {model_dir}.")

    return out_path

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(APP_DIR, "phishing_ensemble_model.pkl"))
    selector = joblib.load(os.path.join(APP_DIR, "selector.pkl"))
    selector_mi = joblib.load(os.path.join(APP_DIR, "selector_mi.pkl"))
    all_cols = pd.read_csv(os.path.join(APP_DIR, "x_columns_only.csv"), nrows=0).columns

    assemble_weights(APP_DIR, "model.safetensors")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(APP_DIR)
    ft_model = AutoModel.from_pretrained(APP_DIR).to(device)
    ft_model.eval()

    return model, selector, selector_mi, all_cols, tokenizer, ft_model, device

model, selector, selector_mi, all_cols, tokenizer, ft_model, device = load_artifacts()

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
    tabular_cols = [c for c in all_cols if not c.startswith(("bert_", "ft_bert_"))]
    df_s = df_s.reindex(columns=tabular_cols, fill_value=0)

    text = [(subject + " " + email_body).strip()]
    inputs = tokenizer(
        text, padding=True, truncation=True, max_length=256, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        b_emb = ft_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()

    b_df = pd.DataFrame(b_emb, columns=[f"ft_bert_{i}" for i in range(b_emb.shape[1])])

    X_raw_combined = pd.concat([df_s.reset_index(drop=True), b_df], axis=1)

    if hasattr(selector, "feature_names_in_"):
        X_raw_combined = X_raw_combined.reindex(
            columns=list(selector.feature_names_in_), fill_value=0
        )
    else:
        X_raw_combined = X_raw_combined.reindex(columns=all_cols, fill_value=0)

    X_var_filtered = selector.transform(X_raw_combined)
    X_sample = selector_mi.transform(X_var_filtered)

    probs = model.predict_proba(X_sample)[0]
    return probs[1]

def check_virustotal_links(urls):
    """Returns (confirmed_malicious, had_error).
    Only flags when at least MIN_MALICIOUS_ENGINES engines say 'malicious'.
    404 (URL never seen) is not an error and not a detection."""
    headers = {"x-apikey": st.secrets["VIRUSTOTAL_API_KEY"]}
    had_error = False

    for url in urls:
        url_id = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
        endpoint = f"https://www.virustotal.com/api/v3/urls/{url_id}"
        try:
            response = requests.get(endpoint, headers=headers, timeout=5)
        except Exception:
            had_error = True
            continue

        if response.status_code == 200:
            stats = (response.json().get("data", {})
                     .get("attributes", {}).get("last_analysis_stats", {}))
            if stats.get("malicious", 0) >= MIN_MALICIOUS_ENGINES:
                return True, had_error
        elif response.status_code != 404:
            had_error = True          # rate limit, bad key, server error

    return False, had_error

MIN_MALICIOUS_ENGINES = 2   # set to 1 while testing, raise later to cut false positives
DOMAIN_RE = re.compile(r"^(?=.{4,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")

def normalize_domain(raw: str):
    d = raw.strip().lower()
    if "@" in d:
        d = d.split("@")[-1]
    d = d.removeprefix("https://").removeprefix("http://").split("/")[0]
    return d if DOMAIN_RE.match(d) else None      # None = not a valid domain

@st.cache_data(ttl=3600, show_spinner=False)
def _vt_domain_lookup(domain: str):
    resp = requests.get(
        f"https://www.virustotal.com/api/v3/domains/{domain}",
        headers={"x-apikey": st.secrets["VIRUSTOTAL_API_KEY"]},
        timeout=5,
    )
    if resp.status_code == 200:
        attrs = resp.json().get("data", {}).get("attributes", {})
        return attrs.get("last_analysis_stats", {})
    if resp.status_code == 404:
        return {}                                  # VirusTotal has never seen it
    raise RuntimeError(f"VirusTotal returned {resp.status_code}")

def check_virustotal_domain(domain: str):
    """Returns (number of engines that flagged it, had_error)."""
    try:
        stats = _vt_domain_lookup(domain)
    except Exception:
        return 0, True
    return stats.get("malicious", 0), False

def defang_domain(domain: str) -> str:
    return domain.replace(".", "[.]")

BRANDS = ["paypal", "microsoft", "google", "amazon", "netflix",
          "facebook", "linkedin", "outlook", "canadapost"]

def lookalike_brand(domain: str):
    """Simple heuristic: the domain's main label resembles a known brand."""
    label = domain.split(".")[-2]
    for brand in BRANDS:
        if label == brand:
            continue
        if brand in label or SequenceMatcher(None, label, brand).ratio() >= 0.8:
            return brand
    return None

PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")

def attack_url(tid: str) -> str:
    return "https://attack.mitre.org/techniques/" + tid.replace(".", "/") + "/"

def defang_url(u: str) -> str:
    return u.replace("http", "hxxp").replace(".", "[.]")

# ---------- MITRE ATT&CK data, loaded from MITRE's STIX 2.1 dataset ----------
ATTACK_STIX_URL = ("https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
                   "master/enterprise-attack/enterprise-attack.json")
STAGE_ORDER = (
    "resource-development",
    "initial-access",
    "defense-evasion",
    "execution"
)
def _clean(text: str, limit: int = 450) -> str:
    text = re.sub(r"\(Citation:[^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.strip().split("\n\n")[0]
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."

@st.cache_data(ttl=7 * 24 * 3600, show_spinner="Loading MITRE ATT&CK data...")
def load_attack_data():
    techs, tactics, course_names, mitigates, version = {}, {}, {}, [], None
    with requests.get(ATTACK_STIX_URL, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = True
        for obj in ijson.items(resp.raw, "objects.item"):      # streams one object at a time
            kind = obj.get("type")
            ref = next((r for r in obj.get("external_references", [])
                        if r.get("source_name") == "mitre-attack"), {})
            if kind == "attack-pattern":
                if (ref.get("external_id")
                        and not obj.get("revoked") and not obj.get("x_mitre_deprecated")):
                    techs[ref["external_id"]] = {
                        "stix_id": obj["id"],
                        "name": obj["name"],
                        "description": _clean(obj.get("description", "")),
                        "phases": [p["phase_name"] for p in obj.get("kill_chain_phases", [])],
                    }
            elif kind == "x-mitre-tactic":
                tactics[obj["x_mitre_shortname"]] = {"name": obj["name"], "id": ref.get("external_id", "")}
            elif kind == "course-of-action":
                course_names[obj["id"]] = obj["name"]
            elif kind == "relationship" and obj.get("relationship_type") == "mitigates":
                mitigates.append((obj["source_ref"], obj["target_ref"]))
            elif kind == "x-mitre-collection":
                version = obj.get("x_mitre_version")
    by_target = {}
    for source, target in mitigates:
        if source in course_names:
            by_target.setdefault(target, set()).add(course_names[source])
    for t in techs.values():
        t["mitigations"] = sorted(by_target.get(t["stix_id"], []))
    return {"techniques": techs, "tactics": tactics, "version": version}


def build_findings(
    text, urls, has_attachment, attachment_type,
    domain, domain_hits, attachment_results=None
):
    findings = []

    # Only map T1583.001 on a confirmed VirusTotal flag, not a lookalike alone.
    if domain and domain_hits >= MIN_MALICIOUS_ENGINES:
        findings.append((
            "T1583.001",
            f"Sender domain {defang_domain(domain)} flagged by "
            f"{domain_hits} engines"
        ))

    risky = [r for r in (attachment_results or [])
        if r["risk"] in ("Medium", "High")]
    if has_attachment and risky:
        names = ", ".join(f"'{r['filename']}' ({r['risk']} risk)" for r in risky)
        findings.append((
            "T1566.001",
            f"Suspicious attachment filename(s): {names}"
        ))

    for result in attachment_results or []:
        if "T1036.007" in result["mitre"]:
            findings.append((
                "T1036.007",
                f"Potential double-extension disguise in "
                f"'{result['filename']}'"
        ))

    if urls:
        shown = ", ".join(defang_url(u) for u in urls[:3])
        findings.append(("T1566.002", f"{len(urls)} link(s): {shown}"))
        findings.append((
            "T1204.001",
            "Potential outcome if the recipient clicks a malicious link"
        ))

    if PHONE_RE.search(text) and re.search(
        r"\b(call|phone|hotline)\b", text, re.I
    ):
        findings.append(("T1566.004", "Phone number plus a call-to-action"))

    return findings


def show_attack_section(findings):
    st.markdown("#### MITRE ATT&CK kill chain")

    try:
        data = load_attack_data()
    except Exception:
        data = None
        st.warning("Couldn't load MITRE ATT&CK data right now. Only technique IDs are shown.")

    if not findings:
        st.info(
            "No specific MITRE ATT&CK technique could be tied to this email's "
            "content (no supported link, attachment, or phone evidence). "
            "The flag came from the overall wording and style of the email."
        )
        t = data["techniques"].get("T1566") if data else None
        if t:
            st.markdown(f"**General context: [T1566 {t['name']}]({attack_url('T1566')})**")
            st.write(t["description"])
            if t["mitigations"]:
                st.caption("MITRE mitigations: " + ", ".join(t["mitigations"]))
            st.caption("Background only. This is not a detected technique for this email.")
        return

    if not data:
        for tid, evidence in findings:
            st.markdown(f"[{tid}]({attack_url(tid)})")
            st.text(evidence)
        return

    techs, tactics = data["techniques"], data["tactics"]
    stages = {s: [] for s in STAGE_ORDER}
    for tid, evidence in findings:                       # stage comes from MITRE's data
        for phase in techs.get(tid, {}).get("phases", []):
            if phase in stages:
                stages[phase].append((tid, evidence))

    active_stages = [s for s in STAGE_ORDER if stages[s]]   # hide empty tactics
    if active_stages:
        cols = st.columns(len(active_stages))
        for col, stage in zip(cols, active_stages):
            tac = tactics.get(stage, {"name": stage.replace("-", " ").title(), "id": ""})
            with col:
                st.markdown(f"**{tac['name']}**  \n`{tac['id']}`")
                for tid, evidence in stages[stage]:
                    st.error(f"[{tid}]({attack_url(tid)}) {techs[tid]['name']}")
                    st.text(evidence)                    # st.text: email-derived text is never rendered as markdown

    st.markdown("**About these techniques (from MITRE ATT&CK)**")
    for tid, _ in findings:
        t = techs.get(tid)
        if t:
            st.markdown(f"**{tid} {t['name']}**")
            st.write(t["description"])
            if t["mitigations"]:
                st.caption("MITRE mitigations: " + ", ".join(t["mitigations"]))
    ver = f" v{data['version']}" if data.get("version") else ""
    st.caption(f"Source: MITRE ATT&CK Enterprise{ver} (STIX 2.1). The technique mapping is inferred from "
               "the email's contents, not confirmed attacker behavior. The Execution stage shows what would "
               "happen if the recipient interacts with the email.")


DANGEROUS_EXTENSIONS = {
    ".exe", ".scr", ".com", ".bat", ".cmd",
    ".ps1", ".vbs", ".vbe", ".js", ".jse",
    ".wsf", ".hta", ".lnk", ".pif", ".msi",
    ".reg", ".dll", ".cpl"
}

ARCHIVE_EXTENSIONS = {
    ".zip", ".rar", ".7z", ".iso", ".img"
}

DECOY_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".jpg", ".jpeg", ".png", ".txt"
}

SUSPICIOUS_KEYWORDS = [
    "urgent", "action required", "payment", "invoice",
    "overdue", "refund", "payroll", "salary", "tax",
    "password", "verify", "account suspended",
    "account locked", "voicemail", "voice message",
    "missed call", "shared document", "secure document",
    "fax", "scan", "delivery", "shipment",
    "purchase order", "remittance"
]


def analyze_attachment_filename(filename):
    filename = filename.strip()
    lower_name = filename.lower()
    base_name = os.path.basename(lower_name)

    result = {
        "filename": filename,
        "score": 0,
        "risk": "Low",
        "indicators": [],
        "recommendation": "",
        "mitre": []
    }

    if not filename:
        return result

    parts = base_name.split(".")
    extensions = ["." + part for part in parts[1:] if part]
    final_extension = os.path.splitext(base_name)[1]

    # 1. Double extension, e.g. invoice.pdf.exe
    if len(extensions) >= 2:
        if (extensions[-2] in DECOY_EXTENSIONS
                and extensions[-1] in DANGEROUS_EXTENSIONS):
            result["score"] += 5
            result["indicators"].append(
                f"Double extension detected: {extensions[-2]} followed by "
                f"{extensions[-1]}. This may disguise the actual file type."
            )
            result["mitre"].append("T1036.007")

    # 2. Executable or script file extension
    if final_extension in DANGEROUS_EXTENSIONS:
        result["score"] += 4
        result["indicators"].append(
            f"Executable or script extension detected: {final_extension}"
        )

    # 3. Archive file
    elif final_extension in ARCHIVE_EXTENSIONS:
        result["score"] += 1
        result["indicators"].append(
            f"Compressed/archive file detected: {final_extension}. "
            "Its contents have not been inspected."
        )

    # 4. Common phishing-related filename wording
    matched_keywords = [
        keyword for keyword in SUSPICIOUS_KEYWORDS
        if keyword in lower_name
    ]

    if matched_keywords:
        result["score"] += min(len(matched_keywords), 2)
        result["indicators"].append(
            "Potential phishing-related wording: "
            + ", ".join(matched_keywords)
        )

    # 5. Unusual filename patterns
    if re.search(r"(.)\1{5,}", base_name):
        result["score"] += 1
        result["indicators"].append(
            "Filename contains an unusually long repeated character sequence."
        )

    if len(filename) > 100:
        result["score"] += 1
        result["indicators"].append("Unusually long filename.")

    # Risk level is a heuristic, not a malware probability.
    if result["score"] >= 5:
        result["risk"] = "High"
        result["recommendation"] = (
            "Treat this attachment as suspicious. Do not open or run it "
            "until its legitimacy has been independently verified."
        )
    elif result["score"] >= 2:
        result["risk"] = "Medium"
        result["recommendation"] = (
            "Verify the sender and confirm that you were expecting this "
            "attachment before opening it."
        )
    else:
        result["risk"] = "Low"
        result["recommendation"] = (
            "No strong filename-based warning was detected. This does not "
            "guarantee that the attachment is safe."
        )

    return result


def show_attachment_results(results):
    if not results:
        return

    st.subheader("Attachment Filename Analysis")

    for result in results:
        st.markdown(f"**{result['filename']}**")

        if result["risk"] == "High":
            st.error(f"High Filename Risk — Score: {result['score']}")
        elif result["risk"] == "Medium":
            st.warning(f"Medium Filename Risk — Score: {result['score']}")
        else:
            st.success(f"Low Filename risk — Score: {result['score']}")

        if result["indicators"]:
            st.markdown("**Indicators Detected:**")
            for indicator in result["indicators"]:
                st.write(f"- {indicator}")
        else:
            st.write("No suspicious filename patterns detected.")

        st.markdown(f"**Recommendation:** {result['recommendation']}")
        st.divider()


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

st.set_page_config(page_title="Phishing Email Detector", page_icon="🎣", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"],
        .stApp {
            background-color: #cbd2dc !important;
        }
        [data-testid="stHeader"] {
            background-color: rgba(203, 210, 220, 0.9) !important;
        }
        [data-testid="stSidebar"] {
            background-color: #eef1f5 !important;
            border-right: 1px solid #b8c1ce !important;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }

        /* Solid white cards with no outline */
        [class*="st-key-card_"] {
            background-color: #ffffff !important;
            border: none !important;
            outline: none !important;
            border-radius: 16px !important;
            padding: 1.25rem 1.4rem !important;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06) !important;
            margin-bottom: 1rem;
        }
        [class*="st-key-card_"] > div,
        [class*="st-key-card_"] [data-testid="stVerticalBlock"],
        [class*="st-key-card_"] [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #ffffff !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }

        /* Inset inputs */
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            background-color: #f3f5f8 !important;
            color: #1f2937 !important;
            border: 1px solid #dfe4ec !important;
            border-radius: 10px !important;
        }
        [data-testid="stSelectbox"] > div > div {
            background-color: #f3f5f8 !important;
            color: #1f2937 !important;
            border-radius: 10px !important;
        }

        /* Required red asterisk label */
        .req-label {
            font-size: 0.875rem;
            font-weight: 400;
            color: #31333f;
            margin-bottom: 0.25rem;
            line-height: 1.4;
        }
        .req-star {
            color: #d93025;
            font-weight: 600;
        }

        /* Empty results placeholder blends directly into the white card */
        .empty-results-box {
            background-color: transparent !important;
            border: none !important;
            padding: 2rem 1rem;
            text-align: center;
            color: #4b5563;
            margin-top: 0.5rem;
        }
        .empty-results-icon {
            font-size: 2rem;
            line-height: 1;
            margin-bottom: 0.6rem;
        }
        .empty-results-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 0.3rem;
        }
        .empty-results-sub {
            font-size: 0.9rem;
            color: #6b7280;
            margin: 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True, key="card_header"):
    st.title("🎣 Phishing Email Detector")
    st.caption(
        "Think your email might be phishing? Input the email's information in the provided fields to find out!"
    )

SAMPLES = {
    "Fake bank alert": "URGENT: Your account suspended. Verify your account now: "
                       "http://192.168.4.7/login or lose access IMMEDIATELY. "
                       "Confirm your password.",
    "Fake invoice": "Hi, please see the attached invoice.zip and click here "
                    "http://bit.ly/3xYz to confirm your payment details.",
    "Normal email": "Hey, are we still on for lunch tomorrow? "
                    "I'll book the table for noon.",
}


def save_and_parse_eml(uploaded_file):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        parsed_subject = (msg.get("subject") or "").strip()

        from_header = msg.get("from") or ""
        parsed_domain = ""
        if "@" in from_header:
            parsed_domain = from_header.split("@")[-1].strip().strip(">").strip()

        body_parts = []
        parsed_attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                disposition = part.get_content_disposition()
                filename = part.get_filename()
                if disposition == "attachment" or filename:
                    if filename:
                        parsed_attachments.append(filename.strip())
                    continue
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    try:
                        body_parts.append(part.get_content())
                    except Exception:
                        pass
                elif ctype == "text/html" and not body_parts:
                    try:
                        html_txt = re.sub(r"<[^>]+>", " ", part.get_content())
                        body_parts.append(re.sub(r"\s+", " ", html_txt).strip())
                    except Exception:
                        pass
        else:
            try:
                body_parts.append(msg.get_content())
            except Exception:
                pass

        parsed_body = "\n".join(p.strip() for p in body_parts if p and p.strip()).strip()
        return parsed_subject, parsed_body, parsed_domain, parsed_attachments
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def req_label(text: str):
    st.markdown(
        f'<div class="req-label">{text} <span class="req-star">*</span></div>',
        unsafe_allow_html=True,
    )


with st.container(border=True, key="card_inputs"):
    st.subheader("Email Inputs")

    input_mode = st.radio(
        "Input Method",
        ["Manual Input", "Drag & Drop"],
        horizontal=True,
        label_visibility="collapsed",
    )

    subject = ""
    email_body = ""
    sender_domain = ""
    eml_attachments = []

    if input_mode == "Manual Input":
        req_label("Enter Email Subject Line")
        subject = st.text_input(
            "Enter Email Subject Line",
            placeholder="e.g., URGENT: Verify your account",
            label_visibility="collapsed",
        )

        req_label("Enter Email Content")
        email_body = st.text_area(
            "Enter Email Content",
            height=220,
            placeholder="Paste the text of the email body here. Emails with little text (< 5 words) will not provide effective analysis.",
            label_visibility="collapsed",
        )

        sender_domain = st.text_input(
            "Sender Email Domain (Optional)",
            placeholder="e.g. paypa1-secure.com",
        )
    else:
        req_label("Upload .eml File")
        uploaded_eml = st.file_uploader(
            "Upload .eml File",
            type=["eml"],
            label_visibility="collapsed",
        )
        if uploaded_eml is not None:
            subject, email_body, sender_domain, eml_attachments = save_and_parse_eml(uploaded_eml)
            st.caption(
                f"Loaded `{uploaded_eml.name}` — Subject: **{subject or '(none)'}** | "
                f"Sender domain: **{sender_domain or '(none)'}** | "
                f"Attachments found: **{len(eml_attachments)}**"
            )

    email = f"Subject: {subject}\n\n{email_body}" if subject else email_body

    contains_attachment = st.toggle(
        "Email contains a File Attachment",
        value=bool(eml_attachments) if input_mode == "Drag & Drop" else False,
    )

    attachment_type = "None"
    attachment_filenames = []

    if contains_attachment:
        default_att_names = "\n".join(eml_attachments) if (input_mode == "Drag & Drop" and eml_attachments) else ""
        default_att_type = "pdf"
        if eml_attachments:
            ext = os.path.splitext(eml_attachments[0])[1].lower().lstrip(".")
            if ext in ["pdf", "docx", "zip", "exe"]:
                default_att_type = ext

        req_label("Provide the file type for one of the Attachment(s)")
        attachment_type = st.selectbox(
            "Provide the file type for one of the Attachment(s)",
            ["pdf", "docx", "zip", "exe"],
            index=["pdf", "docx", "zip", "exe"].index(default_att_type),
            label_visibility="collapsed",
        )

        req_label("Attachment Filenames (One Per Line)")
        filenames_input = st.text_area(
            "Attachment Filenames (One Per Line)",
            value=default_att_names,
            placeholder="Invoice.pdf.exe\nPayment_Confirmation.zip\nReport.docx",
            height=100,
            help="Enter the attachment names exactly as they appear in the email.",
            label_visibility="collapsed",
        )

        attachment_filenames = [
            name.strip()
            for name in filenames_input.splitlines()
            if name.strip()
        ]

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
        req_label("Day of Week Sent")
        day_of_week = st.selectbox(
            "Day of Week Sent",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            label_visibility="collapsed",
        )
    with col2:
        req_label("Hour Sent")
        selected_hour_label = st.selectbox(
            "Hour Sent",
            list(hour_options.keys()),
            label_visibility="collapsed",
        )
        hour = hour_options[selected_hour_label]
    with col3:
        req_label("Device Received On")
        device_type = st.selectbox(
            "Device Received On",
            ["Laptop", "Desktop", "Mobile", "Tablet"],
            label_visibility="collapsed",
        )
    with col4:
        req_label("Email Client Received On")
        email_client = st.selectbox(
            "Email Client Received On",
            ["Thunderbird", "Gmail", "Apple Mail", "Yahoo Mail", "Outlook"],
            label_visibility="collapsed",
        )


def result_bar(p_phish: float, title: str = None):
    p_phish = max(0.0, min(1.0, p_phish))
    phish_pct = p_phish * 100
    legit_pct = 100 - phish_pct
    if title is None:
        title = (f"{phish_pct:.1f}% Phishing Probability" if p_phish >= 0.5
                 else f"{legit_pct:.1f}% Legitimacy Probability")
    st.markdown(f"### {title}")
    st.markdown(
        f"""<div style="display:flex;height:26px;border-radius:8px;overflow:hidden;">
<div style="width:{legit_pct:.1f}%;background:#1e8e3e;"></div>
<div style="width:{phish_pct:.1f}%;background:#d93025;"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:13px;margin-top:4px;">
<span>Legitimate {legit_pct:.1f}%</span><span>Phishing {phish_pct:.1f}%</span>
</div>""",
        unsafe_allow_html=True,
    )


def render_empty_results_placeholder(validation_msg: str = None):
    if validation_msg:
        st.error(validation_msg)
    st.markdown(
        """
        <div class="empty-results-box">
            <div class="empty-results-icon">✉️</div>
            <div class="empty-results-title">No Analysis Performed Yet</div>
            <p class="empty-results-sub">Click <strong>Analyze Email</strong> above to check the email for phishing indicators and view results here.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


REVIEW_THRESHOLD = 0.35   # placeholder; tune after the calibration work

with st.container(border=True, key="card_results"):
    st.subheader("Analysis Results")
    analyze_clicked = st.button("Analyze Email", type="primary")

    if not analyze_clicked:
        render_empty_results_placeholder()
    elif not subject.strip() and not email_body.strip():
        render_empty_results_placeholder("⚠️ Please enter a subject line and email content, and double-check your other inputs!")
    elif not subject.strip():
        render_empty_results_placeholder("⚠️ Please enter a subject line, and double-check your other inputs!")
    elif not email_body.strip():
        render_empty_results_placeholder("⚠️ Please enter email content, and double-check your other inputs!")
    else:
        combined_text = f"{subject} {email_body}"
        extracted_urls = [
            u.rstrip(".,;:)]}>\"'")
            for u in re.findall(r'https?://[^\s]+|www\.[^\s]+', combined_text)
        ]

        attachment_results = [
            analyze_attachment_filename(name)
            for name in attachment_filenames
        ]

        if extracted_urls:
            vt_flagged, link_error = check_virustotal_links(extracted_urls[:5])
        else:
            vt_flagged, link_error = False, False

        domain = normalize_domain(sender_domain) if sender_domain.strip() else None
        if sender_domain.strip() and domain is None:
            st.warning("The provided sender domain doesn't look like a valid domain. The sender-domain check was skipped.")
        domain_hits, domain_error = check_virustotal_domain(domain) if domain else (0, False)
        domain_flagged = domain_hits >= MIN_MALICIOUS_ENGINES

        phishing_prob = classify_email(
            subject, email_body, contains_attachment, attachment_type,
            day_of_week, hour, device_type, email_client
        )

        vt_confirmed = vt_flagged or domain_flagged
        is_flagged = vt_confirmed or phishing_prob >= 0.5

        if vt_confirmed:
            st.error("**Threat detected:** VirusTotal flagged at least one of this email's links or sender domain as malicious.")
            result_bar(1.0, title="Flagged as malicious by VirusTotal")
            if domain_flagged:
                st.error(f"**This sender domain is malicious.** Flagged by {domain_hits} security engines.")
                st.code(defang_domain(domain), language=None)
            st.caption(
                f"The email classifier's independent prediction was "
                f"{phishing_prob * 100:.1f}% phishing. VirusTotal findings takes priority."
            )
        elif phishing_prob >= 0.5:
            st.error("Warning: Potential Phishing Detected")
            result_bar(phishing_prob)
        elif phishing_prob >= REVIEW_THRESHOLD:
            st.warning("Inconclusive: The model leans legitimate but isn't confident. Review this email manually.")
            result_bar(phishing_prob)
        else:
            st.success("No phishing detected by the model")
            result_bar(phishing_prob)
            st.caption("This is not a guarantee the email is safe. The lack of VirusTotal detections doesn't guarantee safety.")

        if domain and not domain_flagged:
            brand = lookalike_brand(domain)
            if brand:
                st.info(f"The sender domain {defang_domain(domain)} resembles '{brand}'. Verify it independently.")

        if contains_attachment:
            if attachment_results:
                show_attachment_results(attachment_results)
            else:
                st.info(
                    "An attachment was reported, but no filename was provided. "
                    "Filename analysis was skipped."
                )

        if domain_error:
            st.warning("VirusTotal couldn't check the sender domain (rate limit or network issue).")
        if link_error:
            st.warning("VirusTotal couldn't check some links (rate limit or network issue).")

        if is_flagged:
            with st.expander("Read More - How was this email analyzed?"):
                findings = build_findings(
                    combined_text,
                    extracted_urls,
                    contains_attachment,
                    attachment_type,
                    domain,
                    domain_hits,
                    attachment_results
                )
                show_attack_section(findings)

with st.sidebar:
    st.header("User Guide")
    st.markdown(
        "1. **Enter Email Subject Line:** Type or paste the email's subject line (required).\n"
        "2. **Enter Email Content:** Paste the full email body and keep any links intact (required; short texts under 5 words may limit analysis).\n"
        "3. **Sender Email Domain (optional):** Enter the sender's email address (e.g., `paypa1-secure.com`) to check reputation and lookalike brands.\n"
        "4. **File Attachment options (optional):** Check *Email contains a File Attachment* to choose a file type (`pdf`, `docx`, `zip`, `exe`) and list attachment filenames (one per line).\n"
        "5. **Email & context details:** Select the *Day of Week Sent*, *Hour Sent*, *Device Received On*, and *Email Client Received On*.\n"
        "6. **Analyze Email:** Click **Analyze Email** to run the phishing check and view results."
    )

    st.divider()
    st.header("Results Analysis")
    st.markdown(
        "- *Virustotal Database Check:* The sender domain and all links in the email are checked against a Virustotal database. If the sender domain or any links are found to be malicious, the email returns as phishing probability 100% every time.\n"
        "- *Phishing Probability:* An email with phishing probability of 50% or more is classified as phishing.\n"
        "- *Filename Analysis:* If attachments and their filenames are provided, filename analysis will provide recommendations depending on if the filename seems malicious or not."
        "- *Read More:* For advanced users. If a email is detected as phishing, the read more dropdown that appears provides more information on phishing techniques."
    )

    st.divider()
    st.header("Disclaimers")
    st.markdown(
        "- *No Guarantees:* Results are informational and do not guarantee an email is safe or malicious. Users should always use their own judgement before interacting with an email.\n"
        "- *External Connections:* External lookups (such as VirusTotal or MITRE data) may be incomplete or unavailable.\n"
        "- *Copy Paste Guidlines:* While pasting text into the tool, avoid clicking on any links from the email or downloading any attachments. These actions are not required to use the detector."
    )
    