# CSI4900-Honor-Project
**Project Title**
Development of an AI-Driven Phishing Detection System Using Advanced NLP and ML Techniques

**Project Overview**
This project is an advanced artificial intelligence (AI) model designed to detect phishing and spam attempts within email communications. The project leverages cutting-edge Natural Language Processing (NLP) techniques alongside a variety of Machine Learning (ML) algorithms to analyze and classify emails. By harnessing large datasets from reputable sources such as Kaggle, the model is meticulously trained and tested to ensure high accuracy and reliability in detecting malicious content. The detection system is implemented with a web application.

**Project Context**
This project was created by two UOttawa Students (David Chen and Ryan Guo) under the supervision of UOttawa Professor Mohamad Hoda.

**Repository File Structure**
- Honor's_Project.ipynb: The primary Jupyter Notebook containing the scripts to  the data pipeline, model training workflows, feature engineering, and evaluation logic.
- app.py: The main Streamlit web application script.
- phishing_email_detection_dataset_original.csv: The original unmodified dataset containing email samples and features from Kaggle
- phishing_email_detection_dataset_model.csv: The cleaned and encoded dataset containing the data that the model trains and tests with directly.
- requirements.txt: Specifies all Python package dependencies and library versions required to run the application locally or on Streamlit Cloud.
- selector.pkl: The serialized variance threshold feature selector used to filter out low-variance features from live inputs.
- selector_mi.pkl: The serialized mutual information feature selector used to reduce the feature space down to the final 100 features expected by the model.
- x_columns_only.csv: A header-only schema file containing just the column names, used by the web app to maintain exact feature alignment without exceeding GitHub file size limits.

**User Guide**
- Open the web application by clicking the provided link: https://ai-phising-detector.streamlit.app/
- Follow the instructions provided on the web application.

