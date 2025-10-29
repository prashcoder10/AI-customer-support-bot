# # from transformers import pipeline
# # from sentence_transformers import SentenceTransformer, util
# # import faiss
# # import numpy as np
# # import csv

# # # Load small generation model
# # gen = pipeline("text-generation", model="distilgpt2", max_new_tokens=60)

# # # Load embedding model
# # embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # # Load FAQ dataset
# # faq_file = "faqs.csv"
# # faqs, faq_embeddings = [], None

# # def load_faqs():
# #     global faqs, faq_embeddings
# #     try:
# #         with open(faq_file, newline='', encoding='utf-8') as f:
# #             reader = csv.DictReader(f)
# #             faqs = [row for row in reader]
# #         corpus = [f["question"] for f in faqs]
# #         faq_embeddings = embedder.encode(corpus, convert_to_tensor=True)
# #     except Exception as e:
# #         print("FAQ load error:", e)

# # load_faqs()

# # def retrieve_faq_answer(query):
# #     if not faqs or faq_embeddings is None:
# #         return "Sorry, FAQ data not loaded.", 0.0
# #     query_emb = embedder.encode(query, convert_to_tensor=True)
# #     scores = util.cos_sim(query_emb, faq_embeddings)[0]
# #     best_idx = int(scores.argmax())
# #     best_score = float(scores[best_idx])
# #     return faqs[best_idx]["answer"], best_score

# # def generate_response(prompt):
# #     out = gen(prompt, num_return_sequences=1)
# #     return out[0]["generated_text"]



# # from transformers import pipeline

# # # Small, fast model
# # generator = pipeline("text-generation", model="distilgpt2")

# # def generate_response(user_message: str) -> str:
# #     """
# #     Generates a reply using a small local GPT model.
# #     """
# #     response = generator(user_message, max_new_tokens=50, do_sample=True, temperature=0.7)
# #     reply = response[0]['generated_text']
# #     # Clean output (remove repeated input text)
# #     return reply[len(user_message):].strip()




# import random
# import re
# from difflib import SequenceMatcher
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# last_topic = None

# FAQS = {
#     "book product": "To book a product, go to the product page, click 'Add to Cart', and proceed to checkout to complete payment.",
#     "track order": "You can track your order using the 'Track Order' option in 'My Account'. Enter your Order ID there.",
#      "book product": "To book a product, go to the product page, click 'Add to Cart', and proceed to checkout to complete payment.",
#     "track order": "You can track your order using the 'Track Order' option in 'My Account'. Enter your Order ID there.",
#     "cancel order": "You can cancel your order before shipment from the 'My Orders' section. Once shipped, cancellation isn't possible.",
#     "return product": "To return a product, go to 'My Orders', select the product, and click 'Return'.",
#     "refund policy": "Refunds are processed within 5–7 business days once the returned product passes quality checks.",
#     "payment issue": "If your payment failed, please try again or use another method. Contact support if the amount was deducted.",
#     "contact support": "You can reach our support team at support@example.com or call 1800-123-4567 (Mon–Fri, 9 AM–6 PM).",
#     "delivery time": "Orders are typically delivered within 3–5 business days depending on your location.",
#     "change address": "You can change your delivery address before dispatch in the 'Address Book' under 'My Account'.",
#     "update profile": "Go to 'My Account' → 'Profile Settings' to update your personal information.",
#     "reset password": "Click 'Forgot Password' on the login page and follow the steps to reset your password.",
#     "login issue": "If you can’t log in, try clearing your browser cache or reset your password.",
#     "offers and discounts": "You can check current offers and coupons under the 'Deals' section of our website.",
#     "order confirmation": "Once payment succeeds, you'll receive an order confirmation email with all details.",
#     "out of stock": "If a product is out of stock, click 'Notify Me' to get an alert when it’s back.",
#     "shipping charges": "Shipping is free for orders above ₹499. Below that, a nominal delivery fee applies.",
#     "multiple items order": "Yes, you can order multiple products together — they’ll be shipped together if available.",
#     "warranty claim": "To claim warranty, go to 'My Orders', select your item, and choose 'Warranty Claim'.",
#     "cancel payment": "If payment was cancelled midway, your order may not have been placed. Check 'My Orders' to confirm.",
#     "invoice request": "Invoices are auto-generated and can be downloaded from 'My Orders'.",
#     "gift wrapping": "You can select gift wrapping at checkout for a small fee.",
#     "international shipping": "Currently, we deliver only within India. International delivery coming soon!",
#     "change payment method": "Payment method cannot be changed once the order is placed.",
#     "replace product": "You can request a replacement within 7 days of delivery under 'My Orders'.",
#     "membership benefits": "Members enjoy early access to sales, exclusive discounts, and free shipping.",
#     "app download": "Download our app from Google Play Store or Apple App Store by searching our brand name.",
#     "feedback": "We’d love your feedback! Share it via the 'Feedback' option on our website footer.",
# }


# SYNONYMS = {
#     "buy": "book product", "purchase": "book product", "order": "book product",
#     "track": "track order", "cancel": "cancel order", "return": "return product",
#     "refund": "refund policy", "payment": "payment issue", "delivery": "delivery time",
#     "shipping": "delivery time", "address": "change address", "profile": "update profile",
#     "password": "reset password", "login": "login issue", "offer": "offers and discounts",
#     "discount": "offers and discounts", "confirmation": "order confirmation",
#     "stock": "out of stock", "fee": "shipping charges", "charges": "shipping charges",
#     "multiple": "multiple items order", "warranty": "warranty claim",
#     "invoice": "invoice request", "gift": "gift wrapping", "international": "international shipping",
#     "replace": "replace product", "member": "membership benefits", "app": "app download",
#     "feedback": "feedback",
# }

# # 🔹 Load Hugging Face Model
# MODEL_NAME = "tiiuae/falcon-7b-instruct"  # or smaller if memory is low
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")

# # Create a pipeline for chat
# llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=200)

# # 🔹 Helper Functions
# def similarity(a, b):
#     return SequenceMatcher(None, a, b).ratio()

# def retrieve_faq_answer(user_query: str):
#     global last_topic
#     query = re.sub(r'[^a-zA-Z0-9 ]', '', user_query.lower())
#     tokens = query.split()

#     # Synonym-based matching
#     for word in tokens:
#         if word in SYNONYMS:
#             mapped_topic = SYNONYMS[word]
#             last_topic = mapped_topic
#             return FAQS[mapped_topic], 1.0

#     # Fuzzy match
#     best_match, best_score = None, 0.0
#     for key, answer in FAQS.items():
#         score = similarity(query, key)
#         if score > best_score:
#             best_score, best_match = answer, score

#     if best_score >= 0.6:
#         last_topic = key
#         return best_match, best_score

#     # Context-based fallback
#     if last_topic and any(word in query for word in ["that", "it", "this", "refund", "return", "cancel"]):
#         return FAQS.get(last_topic, "Can you clarify which topic you're referring to?"), 0.7

#     return None, best_score

# # 🔹 LLM Response using Hugging Face
# def generate_llm_response(prompt: str):
#     try:
#         response = llm_pipeline(f"You are an e-commerce customer support assistant. Answer this query:\n{prompt}", 
#                                 max_length=200, do_sample=True, temperature=0.7)
#         return response[0]['generated_text'].strip()
#     except Exception as e:
#         return f"Sorry, I'm having trouble fetching the answer: {e}"

# # 🔹 Fallback Responses
# def generate_response(prompt: str):
#     fallback_responses = [
#         "Could you please clarify that?",
#         "I'm here to help with orders, returns, or account support!",
#         "Sorry, I didn’t get that — could you rephrase?",
#         "Our team can assist with booking, returns, or payments. What would you like help with?",
#         "Please provide more details about your issue so I can assist better.",
#         "I'm here to help you with order tracking, cancellations, or payments!",
#         "Could you specify your concern — booking, refund, or shipping?",
#     ]
#     return random.choice(fallback_responses)

# # 🔹 Main Bot Response
# def get_bot_response(user_query: str):
#     answer, score = retrieve_faq_answer(user_query)

#     if answer:
#         return answer
#     elif score < 0.4:
#         # Use local Hugging Face LLM
#         llm_reply = generate_llm_response(user_query)
#         return llm_reply or generate_response(user_query)
#     else:
#         return generate_response(user_query)


# # app/llm.py
# import os
# import csv
# import random
# import re
# import numpy as np
# from difflib import SequenceMatcher

# # embeddings + FAISS
# from sentence_transformers import SentenceTransformer, util
# import faiss

# # small local generation model
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# import torch

# # Settings - easy to change
# EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast embeddings
# FAQ_CSV = os.path.join(os.path.dirname(__file__), "..", "faqs.csv")

# # Generation model choice (small by default)
# GEN_MODEL_NAME = "distilgpt2"  # small, runs on CPU reasonably
# # If you have GPU and want a better model, replace with e.g. "gpt2-medium" or a local HF model

# # thresholds
# FAQ_SIM_THRESHOLD = 0.60  # >= this considered a good FAQ match
# ESCALATION_THRESHOLD = 0.35

# # load FAQS (question/answer)
# FAQS = []
# FAQ_QUESTIONS = []
# FAQ_ANSWERS = []

# def load_faqs():
#     global FAQS, FAQ_QUESTIONS, FAQ_ANSWERS
#     path = FAQ_CSV
#     if not os.path.exists(path):
#         print("Warning: FAQ CSV not found at", path)
#         return
#     with open(path, newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             q = row.get("question") or row.get("Question") or ""
#             a = row.get("answer") or row.get("Answer") or ""
#             if q.strip():
#                 FAQS.append({"question": q.strip(), "answer": a.strip()})
#     FAQ_QUESTIONS = [f["question"] for f in FAQS]
#     FAQ_ANSWERS = [f["answer"] for f in FAQS]

# # Embedding index
# embed_model = None
# faq_embeddings = None
# faiss_index = None

# def init_embeddings():
#     global embed_model, faq_embeddings, faiss_index
#     if embed_model is None:
#         embed_model = SentenceTransformer(EMBED_MODEL_NAME)
#     if FAQ_QUESTIONS:
#         # compute embeddings
#         faq_embeddings = embed_model.encode(FAQ_QUESTIONS, convert_to_numpy=True, normalize_embeddings=True)
#         dim = faq_embeddings.shape[1]
#         # build FAISS index
#         faiss_index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine
#         faiss_index.add(faq_embeddings)
#     else:
#         faiss_index = None

# # fuzzy helper
# def similarity(a, b):
#     return SequenceMatcher(None, a, b).ratio()

# def retrieve_faq_answer(query: str):
#     """
#     Returns (answer, score). Score in 0..1 where higher is better.
#     Uses synonyms + fuzzy + semantic search.
#     """
#     q = re.sub(r"[^a-zA-Z0-9 ]", "", query.lower()).strip()
#     # fallback simple fuzzy match first
#     best_score = 0.0
#     best_answer = None

#     for idx, qtext in enumerate(FAQ_QUESTIONS):
#         s = similarity(q, qtext.lower())
#         if s > best_score:
#             best_score = s
#             best_answer = FAQ_ANSWERS[idx]

#     # If semantic index available, use embeddings (more robust)
#     if faiss_index is not None and embed_model is not None:
#         q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
#         D, I = faiss_index.search(q_emb, k=3)
#         if I is not None and len(I) > 0:
#             top_idx = int(I[0][0])
#             score = float(D[0][0])
#             # D contains inner product (cosine) because normalized
#             if score > best_score:
#                 best_score = score
#                 best_answer = FAQ_ANSWERS[top_idx] if top_idx < len(FAQ_ANSWERS) else best_answer

#     return best_answer, float(best_score or 0.0)

# # Generation model (light)
# gen_pipeline = None

# def init_generator(device=None):
#     """
#     Initializes the generation pipeline.
#     device: "cpu" or torch device index (0, "cuda") — we auto-detect if None.
#     """
#     global gen_pipeline
#     if gen_pipeline is not None:
#         return

#     # pick device
#     if device is None:
#         device = 0 if torch.cuda.is_available() else -1

#     try:
#         # Use pipeline for text-generation
#         gen_pipeline = pipeline("text-generation", model=GEN_MODEL_NAME, device=device, tokenizer=GEN_MODEL_NAME)
#     except Exception as e:
#         # last-resort: try to load model/tokenizer explicitly (can be heavy)
#         try:
#             tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
#             model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME)
#             gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
#         except Exception as e2:
#             print("Could not initialize generation pipeline:", e, e2)
#             gen_pipeline = None

# def generate_llm_response(prompt: str, max_new_tokens=120):
#     """
#     Use the local generation pipeline. If unavailable, return a fallback.
#     """
#     if gen_pipeline is None:
#         return fallback_response(prompt)

#     try:
#         # safe prompt: instruct the model to be a support assistant
#         system_prompt = "You are a helpful e-commerce customer support assistant. Answer succinctly."
#         # Many small models don't support chat-format. We'll prepend the instruction:
#         full_prompt = f"{system_prompt}\n\nCustomer: {prompt}\nSupport:"
#         out = gen_pipeline(full_prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.95, num_return_sequences=1)
#         text = out[0]["generated_text"]
#         # Remove the prompt portion if echoed
#         if full_prompt in text:
#             text = text.split(full_prompt, 1)[1]
#         return text.strip()
#     except Exception as e:
#         print("LLM generation error:", e)
#         return fallback_response(prompt)

# def fallback_response(prompt: str):
#     options = [
#         "Could you please clarify that?",
#         "I'm here to help with orders, returns, or account support!",
#         "Sorry, I didn’t get that — could you rephrase?",
#         "Our team can assist with booking, returns, or payments. What would you like help with?",
#         "Please provide more details about your issue so I can assist better.",
#     ]
#     return random.choice(options)

# # High-level bot response combining FAQ + LLM + escalation hint
# def get_bot_response(user_query: str):
#     answer, score = retrieve_faq_answer(user_query)
#     if answer and score >= FAQ_SIM_THRESHOLD:
#         return answer

#     # If near threshold but not confident, produce combined answer + escalate suggestion
#     if answer and score >= ESCALATION_THRESHOLD:
#         # ask LLM to provide a clarification + include FAQ as suggestion
#         llm_out = generate_llm_response(user_query)
#         # prefer LLM if it produced content
#         if llm_out and len(llm_out) > 10:
#             return llm_out
#         return answer

#     # If no FAQ match -> LLM
#     llm_out = generate_llm_response(user_query)
#     return llm_out or fallback_response(user_query)

# # Initialization helper to call at startup
# def init_all(device=None):
#     load_faqs()
#     init_embeddings()
#     init_generator(device=device)



import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

FAQS = []
VECTORIZER = None
FAQ_VECTORS = None
GENERATOR = None

SUPPORT_EMAIL = "support@example.com"  # ✅ update to your real support email


# Load FAQs from CSV safely
def load_faqs(path="app/data/faqs.csv"):
    global FAQS
    try:
        df = pd.read_csv(path, quotechar='"').fillna("")
        df.columns = [c.strip().lower() for c in df.columns]

        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError("CSV must contain 'question' and 'answer' columns.")

        FAQS = df.to_dict(orient="records")
        print(f"FAQs loaded: {len(FAQS)}")
    except Exception as e:
        print("❌ Error loading FAQs:", e)
        FAQS = []


# ✅ Build TF-IDF embeddings
def init_embeddings():
    global VECTORIZER, FAQ_VECTORS
    if not FAQS:
        print("⚠️ No FAQs loaded; cannot create embeddings.")
        return

    VECTORIZER = TfidfVectorizer(stop_words="english", lowercase=True)
    FAQ_VECTORS = VECTORIZER.fit_transform([faq["question"] for faq in FAQS])
    print(" Embeddings initialized.")


# ✅ Initialize fallback generator
def init_generator():
    global GENERATOR
    GENERATOR = pipeline("text-generation", model="gpt2")
    print("✅ Text generator initialized.")


# ✅ Find best matching FAQ
def find_closest_faq(user_query, threshold=0.55):
    if not FAQS or not VECTORIZER:
        return None, 0.0

    query_vec = VECTORIZER.transform([user_query])
    sims = cosine_similarity(query_vec, FAQ_VECTORS)[0]
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    print(f"🔍 Similarity score: {best_score:.2f}")
    if best_score >= threshold:
        return FAQS[best_idx], best_score
    return None, best_score


# ✅ Clean generated reply
def clean_reply(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    reply = lines[-1] if lines else text
    if len(reply.split()) < 3 and len(lines) > 1:
        reply = lines[-2]
    reply = reply.replace("Customer:", "").replace("Support:", "").strip()
    return reply


# Get bot response (FAQ → fallback)
def get_bot_response(user_message, conversation_context=""):
    user_message = user_message.strip()

    # 1️⃣ Try FAQ-based matching first
    faq_match, score = find_closest_faq(user_message)
    if faq_match:
        print(f"📘 Matched FAQ: {faq_match['question']}")
        print(f" Bot reply → {faq_match['answer']}")
        return faq_match["answer"]

    # 2️⃣ Context-aware fallback generation
    system_context = (
        "You are an AI-powered customer support assistant for an online e-commerce platform. "
        "Your role is to help users with queries related to product orders, tracking, returns, payments, "
        "delivery delays, and general shopping assistance. "
        "Always respond politely, clearly, and in a friendly tone. "
        "If a question is unrelated to e-commerce, or if you’re unsure, say: "
        f'“I’m not entirely sure about that. Please contact our support team at {SUPPORT_EMAIL} for further assistance.”'
    )

    prompt = (
        f"{system_context}\n\n"
        f"Recent conversation:\n{conversation_context}\n\n"
        f"Customer: {user_message}\n"
        f"Support:"
    )

    try:
        result = GENERATOR(
            prompt,
            max_length=120,
            num_return_sequences=1,
            temperature=0.7,
            truncation=True,
        )
        raw_text = result[0]["generated_text"]
        reply = clean_reply(raw_text)
    except Exception as e:
        print("⚠️ LLM generation error:", e)
        reply = (
            f"Sorry, I couldn’t process that right now. "
            f"Please contact our support team at {SUPPORT_EMAIL} for further assistance."
        )

    print(f" Bot reply → {reply}")
    return reply
