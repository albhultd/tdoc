from transformers import AutoTokenizer, AutoModelForCausalLM
from docx import Document
from pdfminer.high_level import extract_text
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import re
import pandas as pd  # Import pandas module
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)

EMBEDDING_SEG_LEN = 1500
EMBEDDING_MODEL = "gpt-4" 

EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = "cl100k_base"
ENCODING = "gpt2"

@dataclass
class Paragraph:
    page_num: int
    paragraph_num: int
    content: str

def read_pdf_pdfminer(file_path) -> List[Paragraph]:
    text = extract_text(file_path).replace('\n', ' ').strip()
    paragraphs = batched(text, EMBEDDING_SEG_LEN)
    paragraphs_objs = []
    paragraph_num = 1
    for p in paragraphs:
        para = Paragraph(0, paragraph_num, p)
        paragraphs_objs.append(para)
        paragraph_num += 1
    return paragraphs_objs

def read_docx(file) -> List[Paragraph]:
    doc = Document(file)
    paragraphs = []
    for paragraph_num, paragraph in enumerate(doc.paragraphs, start=1):
        content = paragraph.text.strip()
        if content:
            para = Paragraph(1, paragraph_num, content)
            paragraphs.append(para)
    return paragraphs

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

def batched(iterable, n):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]

def compute_doc_embeddings(df, tokenizer):
    embeddings = {}
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        doc = row["content"]
        doc_embedding = get_embedding(doc, tokenizer)
        embeddings[index] = doc_embedding
    return embeddings

def enhanced_context_extraction(document, keywords, vectorizer, tfidf_scores, top_n=5):
    paragraphs = [para for para in document.split("\n") if para]
    scores = [sum([para.lower().count(keyword) * tfidf_scores[vectorizer.vocabulary_[keyword]] for keyword in keywords if keyword in para.lower()]) for para in paragraphs]

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    relevant_paragraphs = [paragraphs[i] for i in top_indices]
    
    return " ".join(relevant_paragraphs)

def targeted_context_extraction(document, keywords, vectorizer, tfidf_scores, top_n=5):
    paragraphs = [para for para in document.split("\n") if para]
    scores = [sum([para.lower().count(keyword) * tfidf_scores[vectorizer.vocabulary_[keyword]] for keyword in keywords]) for para in paragraphs]

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    relevant_paragraphs = [paragraphs[i] for i in top_indices]
    
    return " ".join(relevant_paragraphs)


def extract_page_and_clause_references(paragraph: str) -> str:
    page_matches = re.findall(r'Page (\d+)', paragraph)
    clause_matches = re.findall(r'Clause (\d+\.\d+)', paragraph)
    
    page_ref = f"Page {page_matches[0]}" if page_matches else ""
    clause_ref = f"Clause {clause_matches[0]}" if clause_matches else ""
    
    return f"({page_ref}, {clause_ref})".strip(", ")

def refine_answer_based_on_question(question: str, answer: str) -> str:
    if "Does the agreement contain" in question:
        if "not" in answer or "No" in answer:
            refined_answer = f"No, the agreement does not contain {answer}"
        else:
            refined_answer = f"Yes, the agreement contains {answer}"
    else:
        refined_answer = answer

    return refined_answer

def answer_query_with_context(question: str, df: pd.DataFrame, top_n_paragraphs: int = 5) -> str:
    question_words = set(question.split())
    
    priority_keywords = ["duration", "term", "period", "month", "year", "day", "week", "agreement", "obligation", "effective date"]
    
    df['relevance_score'] = df['content'].apply(lambda x: len(question_words.intersection(set(x.split()))) + sum([x.lower().count(pk) for pk in priority_keywords]))
    
    most_relevant_paragraphs = df.sort_values(by='relevance_score', ascending=False).iloc[:top_n_paragraphs]['content'].tolist()
    
    context = "\n\n".join(most_relevant_paragraphs)
    prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"
    
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=600)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    references = extract_page_and_clause_references(context)
    answer = refine_answer_based_on_question(question, answer) + " " + references
    
    return answer

def get_embedding(text, tokenizer):
    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state
    except Exception as e:
        print("Error obtaining embedding:", e)
        embedding = []
    return embedding
