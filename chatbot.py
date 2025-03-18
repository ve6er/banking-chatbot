import os
import pickle
import faiss
import numpy as np
import pandas as pd
import re
import string
import random
from typing import List, Dict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib
import ast
from phe import paillier
import dspy
from langchain_community.document_loaders import PyPDFLoader

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
custom_stop_words = stop_words - {"who","what","where","when","why","how","can","do","if","is"}

# Load pre-trained BERT model for sentence embedding
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load retrieval data
retrieval_data = pd.read_csv('retrieval.csv')

## Cleaning Embeddings
def clean_and_convert_embedding(embedding_str):
    cleaned_str = re.sub(r'(?<=\d)\s+(?=-?\d)', ', ', embedding_str)
    return np.array(ast.literal_eval(cleaned_str))

retrieval_data['Embedding'] = retrieval_data['Embedding'].apply(clean_and_convert_embedding)

## Preprocessing Functions
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stop_words]
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text

## Cybersecurity Functions
def generate_keys():
    public_key, private_key = paillier.generate_paillier_keypair()
    return {
        "public_key": str(public_key.n),
        "private_key": private_key
    }

class SecureHomomorphicRedaction:
    def __init__(self, key_parts: Dict[str, str], rbac_roles: Dict[str, List[str]]):
        self.public_key = paillier.PaillierPublicKey(n=int(key_parts['public_key']))
        self.private_key = key_parts['private_key']  # Use the private key directly
        self.rbac_roles = rbac_roles
        self.redaction_counter = 0

    def encrypt(self, value):
        if isinstance(value, str):
            value = sum(ord(char) for char in value)
        return self.public_key.encrypt(value)

    def decrypt(self, encrypted_value, user_role: str, user_id: str):
        if not self._check_rbac(user_role):
            raise PermissionError("User does not have permission to decrypt data")
        return self.private_key.decrypt(encrypted_value)

    def redact(self, original_text, patterns):
        redacted_text = original_text
        redacted_info = []

        for pattern, tag in patterns:
            matches = re.findall(pattern, original_text)
            for match in matches:
                encrypted_value = self.encrypt(match)
                self.redaction_counter += 1
                redaction_id = f"{tag}_{self.redaction_counter}"
                redacted_text = redacted_text.replace(match, f"[REDACTED_{redaction_id}]")
                redacted_info.append({
                    "original_value": match,
                    "encrypted_value": encrypted_value,
                    "type": tag,
                    "redaction_id": redaction_id
                })

        return redacted_text, redacted_info

    def mask(self, original_text, patterns):
        masked_text = original_text
        masked_info = []

        for pattern, tag in patterns:
            matches = re.findall(pattern, original_text)
            for match in matches:
                masked_value = self._generate_masked_value(match, tag)
                masked_text = masked_text.replace(match, masked_value)
                masked_info.append({
                    "original_value": match,
                    "masked_value": masked_value,
                    "type": tag
                })

        return masked_text, masked_info

# Sensitive patterns for redaction
sensitive_patterns = [
    (r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', 'full_name'),       # Full names (e.g., "John Doe")
    (r'\b[A-Z][a-z]+\b(?!\.)', 'name'),                   # First or last names, excluding honorifics
    (r'\b(?:Mrs|Mr|Ms|Dr|Prof)\.\b', 'honorific'),        # Honorifics, which we will choose not to redact
    (r'\b\d{10,13}\b', 'contact_number'),                 # Phone numbers
    (r'\b\d{12,16}\b', 'account_number'),                 # Account numbers
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email'),  # Emails
    (r'\buser\d{3,}\b', 'user_id'),                       # User IDs
    (r'(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*?&]{8,}', 'password')  # Passwords
]
harmful_patterns = [
    r'\bDROP TABLE\b',
    r'\bDELETE\b',
    r'\bINSERT INTO\b',
    r'\bUPDATE\b',
    r'\bSELECT\b',
    r'\b;--\b',
    r'\b;#\b',       # SQL comment
    r'\b\' OR \'1\'=\'1\'\b',  # SQL injection
    r'\bUNION SELECT\b',
    r'\bEXEC\b',
    r'\bEXECUTE\b',
    r'\bALTER\b',
    r'\bCREATE\b',
    r'\b/\*\b',  # Matches the start of block comments
    r'\b\*/\b',
    r'\b(?:cmd|powershell|exec|system)\s*\((.*?)\)',  # Command injection
    r'\b(drop\s+table|insert\s+into|delete\s+from|select\s+.*\s+from|update\s+set|union\s+select|exec|execute|truncate)\b',  # SQL Injection
    r'\b(=|--|;|#|/\*|SELECT|FROM|WHERE|INSERT|DELETE|UPDATE|DROP|TRUNCATE|EXEC|EXECUTE)\b'  # Common SQL commands
]

def filter_harmful_input(user_question):
    harmful_found = True  # Initialize as True to enter the loop
    input_text = user_question.replace("/*", "").replace("*/", "").replace("\'", "")

    while harmful_found:
        harmful_found = False  # Reset harmful_found for this iteration

        # Check for any harmful pattern
        for pattern in harmful_patterns:
            if re.search(pattern, input_text, flags=re.IGNORECASE):
                harmful_found = True  # Set to True if any harmful pattern is found
                break  # No need to check further patterns

        if harmful_found:
            # Check if "--" is in the input text
            if re.search(r'--', input_text):
                # Remove harmful patterns if "--" is found
                for pattern in harmful_patterns:
                    input_text = re.sub(pattern, '', input_text, flags=re.IGNORECASE)
                # Remove '--' as well
                input_text = re.sub(r'--', '', input_text, flags=re.IGNORECASE)
            else:
                # Remove everything from the start to the first semicolon if no "--" found
                #input_text = re.sub(r'^.*?;', '', input_text, flags=re.DOTALL)
                input_text = re.sub(pattern, '', input_text, flags=re.DOTALL)

    return input_text.strip()

def preprocess_query(user_question, homomorphic_scheme):
    # Step 1: Filter harmful input
    filtered_query = filter_harmful_input(user_question)
    
    # cleaned_query = preprocess_text(filtered_query)

    # Step 2: Redact and mask sensitive information
    redacted_query, redacted_info = homomorphic_scheme.redact(filtered_query, sensitive_patterns)
    masked_query, masked_info = homomorphic_scheme.mask(filtered_query, sensitive_patterns)

    # Step 3: Clean the filtered query using text preprocessing

    return {
        "sanitized_query": user_question,
        "filtered_query": filtered_query,
        "redacted_query": redacted_query,
        "masked_query": masked_query,
        "redacted_info": redacted_info,
        "masked_info": masked_info,
        # "cleaned_query": cleaned_query  # Add cleaned query to the return dictionary
    }

## Intent and Domain Classification
vectorizer = joblib.load('joblib_files/tfidf_vectorizer_domain.joblib')
svm_domain_classifier = joblib.load('joblib_files/svm_domain_classifier.joblib')
intent_pipeline = joblib.load('joblib_files/intent_pipeline.joblib')

def classify_query(cleaned_query):
    query_tfidf = vectorizer.transform([cleaned_query])
    domain_prediction = svm_domain_classifier.predict(query_tfidf)[0]
    return domain_prediction

def classify_intent(cleaned_query):
    intent_prediction = intent_pipeline.predict([cleaned_query])
    return intent_prediction[0]

def find_similar_question_bert(cleaned_query, retrieval_data, threshold=0.85):
    user_embedding = model.encode([cleaned_query])[0]
    retrieval_data['Similarity'] = retrieval_data['Embedding'].apply(
        lambda emb: cosine_similarity([user_embedding], [emb])[0][0]
    )
    top_similar = retrieval_data.nlargest(1, 'Similarity').iloc[0]
    if top_similar['Similarity'] < threshold:
        return None
    return top_similar['Answer']

def handle_user_query(user_question, homomorphic_scheme):
    preprocessed_query_info = preprocess_query(user_question, homomorphic_scheme)
    nlp_filtered_query = preprocessed_query_info['masked_query']
    nlp_filtered_query = preprocess_text(nlp_filtered_query)
    domain_prediction = classify_query(nlp_filtered_query)
    if domain_prediction:
        answer = find_similar_question_bert(nlp_filtered_query, retrieval_data)
        intent = classify_intent(nlp_filtered_query)
        return {
            "domain": "Banking",
            "masked_query": nlp_filtered_query,
            "intent": intent
        }
    else:
        return {
            "domain": "nonBanking",
            "masked_query": nlp_filtered_query,
            "intent": "Not Banking"
        }

def apply_filters(user_input):
    key_parts = generate_keys()
    rbac_roles = {"decrypt": ["admin", "security_officer"]}
    homomorphic_scheme = SecureHomomorphicRedaction(key_parts, rbac_roles)
    user_question = user_input
    response = handle_user_query(user_question, homomorphic_scheme)
    intent = response["intent"]
    masked_query = response["masked_query"]
    return masked_query, intent

## FAISS Retriever Setup
db_path = "db5/"
if not os.path.exists(db_path):
    os.makedirs(db_path)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index_path = os.path.join(db_path, "faiss_index5.bin")
docs_path = os.path.join(db_path, "docs.pkl")

# Load multiple PDFs
pdf_files = [
    'PDFS/Banking  XI  class.pdf',
    'PDFS/BANKPolicies.pdf',
    'PDFS/SBAA1303.pdf',
    'PDFS/the-indian-banking-system.pdf',
]

docs = []
ids = []
i = 0

# Load documents from PDFs
for pdf_file in pdf_files:
    pdf_loader = PyPDFLoader(pdf_file)
    documents = pdf_loader.load_and_split()
    for doc in documents:
        docs.append(doc.page_content)
        ids.append(str(i))
        i += 1
        
# Load data from CSV files
csv_files = [
    'CSV/BankFAQs.csv',
    'CSV/hugging_face_banking.csv'
]

# Define a function to chunk long text
def chunk_text(text, max_chunk_size=500):
    """Split text into chunks of a maximum size."""
    chunks = []
    while len(text) > max_chunk_size:
        chunk = text[:max_chunk_size]
        chunks.append(chunk)
        text = text[max_chunk_size:]
    if text:
        chunks.append(text)
    return chunks

for csv_file_path in csv_files:
    csv_data = pd.read_csv(csv_file_path)
    question_column = 'Question' if 'BankFAQs' in csv_file_path else 'instruction'
    answer_column = 'Answer' if 'BankFAQs' in csv_file_path else 'response'

    for index, row in csv_data.iterrows():
        question = row[question_column]
        answer = row[answer_column]
        
        # Chunk both question and answer
        question_chunks = chunk_text(question)
        answer_chunks = chunk_text(answer)
        
        # Append chunked Q&A pairs to docs
        for q_chunk in question_chunks:
            for a_chunk in answer_chunks:
                docs.append(f"Q: {q_chunk} A: {a_chunk}")
                ids.append(str(i))
                i += 1

# Check if FAISS index and documents already exist
if os.path.exists(faiss_index_path) and os.path.exists(docs_path):
    faiss_index = faiss.read_index(faiss_index_path)
    with open(docs_path, "rb") as f:
        stored_docs = pickle.load(f)
else:
    # Generate embeddings for all documents
    doc_embeddings = embedding_model.encode(docs)

    # Initialize a FAISS index and add embeddings
    embedding_dim = doc_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dim)  # L2 (Euclidean) distance
    faiss_index.add(np.array(doc_embeddings))

    # Save the FAISS index and document texts for later use
    faiss.write_index(faiss_index, faiss_index_path)
    with open(docs_path, "wb") as f:
        pickle.dump({"docs": docs, "ids": ids}, f)

class FAISSRetriever:
    def __init__(self, faiss_index, docs, k=5):
        self.index = faiss_index
        self.docs = docs
        self.k = k

    def retrieve(self, query):
        query_embedding = embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding, self.k)
        return [self.docs[i] for i in indices[0]]

retriever_model = FAISSRetriever(faiss_index, docs)

## RAG Module
lm = dspy.OllamaLocal(model='llama3.2')
dspy.settings.configure(lm=lm)

class RAG(dspy.Module):
    def __init__(self, retriever, num_passages=3):
        super().__init__()
        self.retrieve = retriever
        self.generate_answer = dspy.ChainOfThought("context, question, intent -> answer")
        self.history = []
        self.num_passages = num_passages

    def forward(self, question, intent):  
        recent_history = self.history[-3:]
        context_from_history = ""
        if recent_history:
            last_entry = recent_history[-1]
            weighted_context = f"Q: {last_entry['question']} A: {last_entry['answer']} " * 5
            previous_context = " ".join(
                [f"Q: {entry['question']} A: {entry['answer']}" for entry in recent_history[:-1]]
            )
            context_from_history = f"{weighted_context} {previous_context}"

        combined_query = f"{context_from_history} Question: {question}"
        
        # Retrieve context relevant to the query
        context_passages = self.retrieve.retrieve(combined_query)
        unique_context = list(dict.fromkeys(context_passages))[:self.num_passages]
        
        context_str = " ".join(unique_context)

        # Modify the prompt to include both the intent and the context
        prompt = (
            f"As a banking chatbot, your role is to provide helpful, concise, and clear answers to customer questions about topics such as Card Management, Transaction Inquiries, Loans and Mortgages, Transfers and Payments, Customer Service, Fees and Charges, Identity Verification, Account Management, and general Support."
            f"If the intent is identified as 'Not Banking,' kindly redirect the user to ask questions only about banking topics."
            f"Your task is to provide a clear and accurate response based on the provided context and the user's query intent. Please pay attention to the intent and provide a professional answer accordingly."
            f"Do not include any reasoning, and avoid unnecessary explanations.\n\n"
            f"Intent: {intent}\n\n"
            f"Context: {context_str}\n\n"
            f"Question: {question}\nAnswer:"
        )
        
        # Ensure intent is passed as part of the input to the model
        prediction = self.generate_answer(
            context=context_str, 
            question=question, 
            intent=intent,  # Ensure intent is passed explicitly as well
            prompt=prompt
        )
        self.history.append({"question": question, "answer": prediction.answer})  
        
        return prediction.answer

class SecureBankingChatbot:
    def __init__(self, retriever):
        self.rag = RAG(retriever)

    def ask(self, user_input):
        masked_query, intent = apply_filters(user_input)
        if intent:
            response = self.rag(masked_query, intent)
        else:
            response = "Sorry, I couldn't understand your request. Could you clarify?"
        return response

chatbot = SecureBankingChatbot(retriever_model)