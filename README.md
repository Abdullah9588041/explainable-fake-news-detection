# Explainable Fake News Detection using BERT

This repository presents an end-to-end fake news detection system based on a fine-tuned
Bidirectional Encoder Representations from Transformers (BERT) model.
Beyond achieving strong predictive performance, the proposed system emphasizes
model transparency by integrating Explainable Artificial Intelligence (XAI) techniques,
namely LIME and SHAP, to provide word-level interpretations of predictions.

The project is designed as a complete research and deployment pipeline, including
data analysis, model training, evaluation, explainability, and real-world deployment.

---

## Key Contributions
- Fine-tuning of a BERT-based model for binary fake news classification
- Comprehensive evaluation using accuracy, precision, recall, and F1-score
- Integration of SHAP for token-level contribution analysis
- Integration of LIME for local, instance-level explanations
- Deployment of the trained model as an interactive web application on Hugging Face Spaces

---

## Dataset
The experiments are conducted on a preprocessed short-text fake news dataset provided in CSV format:
- `shorttextpreprocessedtrain.csv`
- `shorttextpreprocessedtest.csv`

Each record in the dataset contains:
- `text`: cleaned and normalized news text
- `label`: binary class label (0 = Fake News, 1 = Real News)

The dataset exhibits class imbalance, reflecting real-world misinformation patterns,
and is suitable for evaluating both classification performance and explainability methods.

---

## Model and Methodology
The classification model is based on `bert-base-uncased` with a sequence classification head.
Texts are tokenized using the BERT tokenizer with truncation and padding, and the model
is fine-tuned using cross-entropy loss and the AdamW optimizer.

Evaluation is performed on a held-out test set, where the model achieves approximately
**91% accuracy**, with detailed class-wise performance analysis reported in the project notebooks
and accompanying IEEE-format report.

---

## Explainability
To improve interpretability and user trust, the project integrates two complementary
explainability techniques:
- **SHAP**: Provides token-level contribution analysis based on Shapley value approximations
- **LIME**: Generates local explanations by identifying words that most influence individual predictions

For deployment compatibility, LIME explanations are rendered as a lightweight tabular
visualization highlighting influential words and their contribution weights.

---

## Deployment
The trained model is deployed as an interactive web application using Hugging Face Spaces.
The application allows users to input news text, view predicted probabilities, and generate
LIME-based explanations in real time.

**Live Demo:**  
👉 https://huggingface.co/spaces/AbdullahBinAjmal/fake-news-bert

---

## Project Structure
- `data/` – preprocessed training and test datasets  
- `notebooks/` – Jupyter notebooks for data analysis, training, evaluation, SHAP, and LIME  
- `results/` – evaluation metrics, confusion matrix, and explanation visualizations  
- `report/` – IEEE-format LaTeX source and compiled PDF  

---

## Technologies Used
- Python  
- PyTorch  
- Hugging Face Transformers  
- LIME  
- SHAP  
- Gradio  

---

## Author
**Abdullah**  
MS Data Science  
Pak-Austria Fachhochschule: Institute of Applied Sciences and Technology  
