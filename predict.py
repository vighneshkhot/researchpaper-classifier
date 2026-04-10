import pickle
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Load everything from saved model folder ──────────────────
save_path     = "model/"   # change this path if your folder is elsewhere
tokenizer     = AutoTokenizer.from_pretrained(save_path)
model         = AutoModelForSequenceClassification.from_pretrained(save_path)
label_encoder = pickle.load(open(save_path + "label_encoder.pkl", "rb"))
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
import pickle
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SAVE_PATH = "model/"

# Load model
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
model = AutoModelForSequenceClassification.from_pretrained(SAVE_PATH)

with open(SAVE_PATH + "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

MAX_LENGTH = 128

def clean_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_category(text):
    cleaned = clean_text(text)
    if cleaned == "":
        return "Invalid input"

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred_idx = torch.argmax(outputs.logits, dim=1).item()

    return label_encoder.inverse_transform([pred_idx])[0]


# ✅ Interactive testing
if __name__ == "__main__":
    print("✅ Model loaded successfully!")
    print("Type 'exit' to quit\n")

    while True:
        text = input("Enter abstract: ")

        if text.lower() == "exit":
            break

        result = predict_category(text)
        print("👉 Predicted Category:", result)
        print("-" * 50)

MAX_LENGTH = 128

# ── Clean text function ───────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ── Prediction function ───────────────────────────────────────
def predict_category(abstract_text):
    cleaned = clean_text(abstract_text)
    if cleaned == "":
        return "Invalid input — empty text"
    inputs = tokenizer(
        cleaned,
        return_tensors = "pt",
        padding        = "max_length",
        truncation     = True,
        max_length     = MAX_LENGTH
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs  = model(**inputs)
        pred_idx = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([pred_idx])[0]

# ── Test with sample abstracts ────────────────────────────────
if __name__ == "__main__":
    samples = [
        "This paper proposes a new neural network architecture for image classification using convolutional layers and attention mechanisms.",
        "We study the distribution of prime numbers and prove a new bound on the Riemann zeta function.",
        "This paper analyzes stock market volatility using Bayesian statistical methods and time series models.",
        "We present new observations of gravitational waves from binary black hole mergers detected by LIGO."
    ]

    print("Model loaded successfully!")
    print(f"Device     : {device}")
    print(f"Categories : {len(label_encoder.classes_)} total")
    print("\n--- Predictions ---")

    for i, abstract in enumerate(samples):
        category = predict_category(abstract)
        print(f"\nPaper {i+1}: {abstract[:80]}...")
        print(f"Predicted : {category}")

        