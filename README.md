# CSI4900-Honor-Project
Repository containing the code and files for the phishing AI model and web app.

PROJECT ROADMAP

Phase 1: Data and ML (partner)
✅ Dataset, DistilBERT embeddings (frozen) + metadata, Logistic Regression / Random Forest / soft-voting ensemble
⬜ Extra models: XGBoost, Naive Bayes + TF-IDF, MLP as the deep neural network, Sentence-BERT for RAG

Phase 2: App
✅ Streamlit demo (rule-based prototype) with verdict, risk score, signals, links, ATT&CK tag, severity
✅ GitHub repo with .gitignore, tests, CI workflow (confirm the Actions run is green)
⬜ .eml upload with SPF/DKIM/DMARC and From vs. Reply-To parsing

Phase 3: Security layer
⬜ VirusTotal and AbuseIPDB enrichment (keys in .env, mocked in tests)
⬜ Severity score from model score plus intel hits, with a disposition field
⬜ Elasticsearch + Kibana logging every scan, with one Sigma rule in Elastic Security
⬜ Shuffle playbook: webhook in, enrichment, branch, alert, write back
⬜ RAG stage 1: similar-email retrieval with Sentence-BERT, using Elasticsearch as the vector store
⬜ RAG stage 2 (optional): LLM explanation grounded in a small ATT&CK knowledge base, with prompt-injection guardrails

Phase 4: Infrastructure and DevSecOps
⬜ FastAPI wrapper around src/ (API key auth, input size limit)
⬜ Dockerfile and docker-compose.yml (app, API, Elasticsearch, Kibana, Shuffle)
⬜ Ubuntu VM built with Ansible (Docker install, firewall, key-only SSH, secrets, start the stack)
⬜ Terraform: S3 bucket, least-privilege IAM role, Secrets Manager
⬜ CI: GitHub Actions (pytest, ruff, Bandit, pip-audit, Trivy)
⬜ CD: Actions builds the image and pushes to ghcr.io, and Ansible pulls and runs it

Phase 5: Unlikely or Considerations
⬜ OWASP ZAP scan of your own app, fix findings, write up before and after
⬜ GoPhish test campaigns (own inboxes only), which also produce real .eml samples
⬜ TypeScript browser extension that sends highlighted text to the FastAPI endpoint (stretch), with npm audit and ESLint in CI
⬜ Nmap check of your own VM
⬜ Evasion testing write-up (typos, homoglyphs, lookalike domains, shorteners)
