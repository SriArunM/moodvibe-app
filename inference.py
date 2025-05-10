import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import pandas as pd

# Define device (use GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the path to the saved model and tokenizer
model_path = "./stress_model"  # Adjust this to the directory containing your files

# Load the label encoder
with open(f"{model_path}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)

# Load the model
model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.to(device)
model.eval()  # Set model to evaluation mode

# Function to preprocess and predict
def predict_stress(statement):
    # Tokenize the input statement
    inputs = tokenizer(
        statement,
        max_length=128,  # Adjust if different during training
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Move inputs to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    
    # Map the predicted class to the label using the label encoder
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label

# Example usage
test_statements = [
    "I feel overwhelmed and can't sleep at night.",
    "Everything is going great in my life!",
    "I feel sad and hopeless all the time."
]

for statement in test_statements:
    prediction = predict_stress(statement)
    print(f"Statement: {statement}")
    print(f"Predicted Status: {prediction}\n")

# Optional: Batch inference on a dataset
def predict_batch(dataframe):
    predictions = []
    for statement in dataframe["statement"]:
        pred = predict_stress(statement)
        predictions.append(pred)
    return predictions

# Example for batch inference (uncomment to use)
# df = pd.read_csv("your_test_dataset.csv")  # Load your test dataset
# df["predicted_status"] = predict_batch(df)
# print(df[["statement", "predicted_status"]])