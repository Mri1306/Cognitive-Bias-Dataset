# Cognitive-Bias-Dataset-Generator
This project generates synthetic dialogue datasets that simulate cognitive biases in human decision-making. It leverages Google Gemini (for question generation) and Groq LLM (for biased responses) to create structured datasets with questions, AI suggestions, biased answers, and confidence scores.

🚀 Features

✅ Generates unique decision-making questions across diverse domains (finance, health, career, relationships, etc.)

✅ Provides AI suggestions that may subtly influence participant answers

✅ Simulates 33+ cognitive biases (e.g., Anchoring, Confirmation Bias, Authority Bias, etc.)

✅ Assigns confidence scores tailored to each bias type

✅ Handles API quota errors with key rotation and fallback questions

✅ Saves outputs in:

A combined JSON dataset

Individual JSON files for each participant

📂 Project Structure
.
├── DG.py                       # Main script
├── Cognitive_Dataset/          # Folder with per-participant datasets (generated after running)
├── cognitive_bias_dataset.json # Combined dataset file
├── .env                        # Store your API keys here
└── README.md                   # Project documentation

By default, it generates:

5 participants, each answering 60 questions.

Saves output to cognitive_bias_dataset(41-45).json + individual files in Cognitive_Dataset/.

You can modify parameters in main() inside DG.py:

NUM_PARTICIPANTS = 10
QUESTIONS_PER_PARTICIPANT = 50

📊 Output Example

Each participant file (e.g., P01.json) looks like this:

{
  "participant_id": "P01",
  "dialogue": [
    {
      "question_id": "Q01",
      "question": "Should I invest in cryptocurrency right now?",
      "ai_suggestion": "AI suggests digital currencies have shown massive growth potential in recent years.",
      "participant_answer": "Cryptocurrency seems like the future; I’d definitely invest now.",
      "bias_type": "Anchoring",
      "confidence": 82
    }
  ]
}

📌 Applications

Cognitive psychology research

Bias-aware AI training

Educational datasets for ML models

Synthetic data generation for NLP
