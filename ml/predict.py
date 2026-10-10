
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "career_model.joblib"


def predict_career(answers):
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model not found. Run train_model.py first."
        )

    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    features = saved["features"]

    # Check that every feature is provided.
    missing = [name for name in features if name not in answers]
    if missing:
        raise ValueError(f"Missing answers: {missing}")

    # Ratings must be integers from 1 to 5.
    for name in features:
        value = answers[name]
        if not isinstance(value, (int, float)) or not 1 <= value <= 5:
            raise ValueError(f"{name} must be a rating from 1 to 5.")

    student = pd.DataFrame(
        [[answers[name] for name in features]],
        columns=features,
    )

    prediction = model.predict(student)[0]
    probabilities = model.predict_proba(student)[0]

    ranking = sorted(
        zip(model.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "predicted_career": prediction,
        "model_probabilities": [
            {
                "career": career,
                "score": round(float(probability) * 100, 2),
            }
            for career, probability in ranking
        ],
    }


if __name__ == "__main__":
    sample_student = {"programming": 2, "mathematics": 2, "creativity": 5,
 "communication": 4, "design": 5, "security": 1,
 "gaming": 2, "problem_solving": 2}
    
    result = predict_career(sample_student)

    print("\nPredicted career:", result["predicted_career"])
    print("\nCareer ranking:")

    for item in result["model_probabilities"]:
        print(f'{item["career"]}: {item["score"]}%')
