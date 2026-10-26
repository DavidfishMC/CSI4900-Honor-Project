import re

URGENT = ["urgent", "verify now", "verify your account", "account suspended",
          "act now", "immediately", "click here", "password",
          "confirm your", "limited time", "unusual activity"]
SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly"]
ATTACH_HINTS = ["attached", "attachment", ".exe", ".zip", ".docm", ".xlsm", ".iso"]
URL_RE = re.compile(r"https?://[^\s<>\"')]+", re.I)


def load_model():
    return None  # later: joblib.load(...) for the trained model


def analyze(text: str) -> dict:
    t = text.lower()
    urls = URL_RE.findall(text)
    signals, score = {}, 0

    hits = [w for w in URGENT if w in t]
    if hits:
        signals["Urgency / credential language"] = ", ".join(hits)
        score += min(len(hits), 4) * 15

    if urls:
        signals["Links in email"] = len(urls)
        score += 10
    ip_urls = [u for u in urls if re.match(r"https?://\d{1,3}(\.\d{1,3}){3}", u)]
    if ip_urls:
        signals["Link uses a raw IP address"] = ip_urls[0]
        score += 25
    short = [u for u in urls if any(s in u.lower() for s in SHORTENERS)]
    if short:
        signals["URL shortener used"] = short[0]
        score += 15

    caps = [w for w in re.findall(r"\b[A-Z]{4,}\b", text)]
    if len(caps) >= 3:
        signals["Excessive ALL-CAPS words"] = len(caps)
        score += 10

    attach = [a for a in ATTACH_HINTS if a in t]
    if attach:
        signals["Attachment references"] = ", ".join(attach)
        score += 10

    score = min(score, 100)
    label = "phishing" if score >= 40 else "safe"
    severity = "High" if score >= 70 else "Medium" if score >= 40 else "Low"

    if label == "phishing":
        if attach:
            tag = "T1566.001 Spearphishing Attachment"
        elif urls:
            tag = "T1566.002 Spearphishing Link"
        else:
            tag = "T1566 Phishing"
    else:
        tag = "None"

    return {"label": label, "score": score, "severity": severity,
            "attack_tag": tag, "signals": signals, "urls": urls}


def predict(model, text: str):
    """Keeps the (label, confidence) contract the pytest tests use."""
    r = analyze(text)
    return r["label"], r["score"] / 100