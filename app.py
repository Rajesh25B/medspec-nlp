import gradio as gr
import torch
import pickle

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
device = torch.device('cpu')  # CPU for deployment


def load_artifacts():
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained('medicalai/ClinicalBERT')

    model = AutoModelForSequenceClassification.from_pretrained(
        'medicalai/ClinicalBERT',
        num_labels=8
    )
    model.load_state_dict(
        torch.load('models/best_model.pt', map_location=device)
    )
    model.eval()
    return model, tokenizer, le

model, tokenizer, le = load_artifacts()


# Prediction function 
def predict_specialty(clinical_note):
    if not clinical_note.strip():
        return "Please enter a clinical note.", {}, ""

    encoding = tokenizer(
        clinical_note,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_label = torch.argmax(probs).item()
        confidence = probs[pred_label].item()

    specialty = le.inverse_transform([pred_label])[0]

    # All specialties with probabilities for label output
    all_probs = {
        le.inverse_transform([i])[0]: round(probs[i].item(), 4)
        for i in range(len(probs))
    }

    summary = f"Predicted: **{specialty}** with {confidence*100:.2f}% confidence"

    return specialty, all_probs, summary


# Sample clinical notes
examples = [
    ["Patient presents with crushing substernal chest pain radiating to the left arm. ECG shows ST elevation in V1-V4. Troponin elevated. Diagnosed with STEMI. Initiated dual antiplatelet therapy and urgent PCI planned."],
    ["MRI of the lumbar spine reveals a herniated disc at L4-L5 with significant nerve root compression. Patient reports radiating pain down the right leg. Surgical consultation recommended."],
    ["Chest X-ray demonstrates bilateral infiltrates. CT abdomen shows 2cm hypodense liver lesion. Follow-up MRI advised. No pleural effusion identified."],
    ["Patient presents with hematuria and flank pain. Ultrasound reveals a 6mm stone in the right ureter. Cystoscopy and ureteroscopy planned for stone removal."],
    ["Upper endoscopy reveals a 1cm gastric ulcer with active bleeding. H. pylori positive. Started on proton pump inhibitor and triple therapy antibiotic regimen."]
]


# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="MedSpec NLP") as demo:

    gr.Markdown("""
    # MedSpec NLP — Medical Specialty Classifier
    ### Fine-tuned ClinicalBERT for Medical Transcription Classification
    Paste any clinical note or transcription to predict the medical specialty.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Clinical Note / Transcription",
                placeholder="Paste clinical transcription here...",
                lines=10
            )
            predict_btn = gr.Button("Predict Specialty", variant="primary")

        with gr.Column(scale=1):
            specialty_output = gr.Textbox(
                label="Predicted Specialty",
                interactive=False
            )
            summary_output = gr.Markdown()
            prob_output = gr.Label(
                label="Confidence Scores (All Specialties)",
                num_top_classes=8
            )

    gr.Markdown("### Try these examples:")
    gr.Examples(
        examples=examples,
        inputs=text_input,
        label="Sample Clinical Notes"
    )

    gr.Markdown("""
    ---
    ### Model Performance
    | Model | Accuracy |
    |---|---|
    | Baseline (TF-IDF + Logistic Regression) | 38.00% |
    | DistilBERT | 63.55% |
    | **ClinicalBERT (deployed)** | **68.59%** |

    **Dataset:** MTSamples — 3823 clinical transcriptions across 8 specialties
    **Specialties:** Surgery, Cardiovascular/Pulmonary, Orthopedic, Radiology,
    Neurology, Gastroenterology, Urology, Obstetrics/Gynecology
    """)

    predict_btn.click(
        fn=predict_specialty,
        inputs=text_input,
        outputs=[specialty_output, prob_output, summary_output]
    )

# ── Launch ─────────────
if __name__ == "__main__":
    demo.launch()
