import torch
import pickle

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model(model_path, encoder_path, device):
    with open(encoder_path, 'rb') as f:
        le = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained('medicalai/ClinicalBERT')

    model = AutoModelForSequenceClassification.from_pretrained(
        'medicalai/ClinicalBERT',
        num_labels=8
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, tokenizer, le


def predict_specialty(text, model, tokenizer, le, device):
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    specialty = le.inverse_transform([pred_label])[0]

    top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
    top3 = [
        (le.inverse_transform([idx.item()])[0], prob.item())
        for idx, prob in zip(top3_indices[0], top3_probs[0])
    ]

    return specialty, confidence, top3


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, le = load_model('models/best_model.pt', 'models/label_encoder.pkl', device)

    text = input("Enter clinical note: ")
    specialty, confidence, top3 = predict_specialty(text, model, tokenizer, le, device)

    print(f"\nPredicted Specialty : {specialty}")
    print(f"Confidence          : {confidence*100:.2f}%")
    print(f"\nTop 3 Predictions:")
    
    for rank, (spec, prob) in enumerate(top3, 1):
        print(f"  {rank}. {spec:<35} {prob*100:.2f}%")

