
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "career_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "career_model.joblib"

FEATURES = [
    "programming",
    "mathematics",
    "creativity",
    "communication",
    "problem_solving",
    "design",
    "security",
    "gaming",
]

CAREERS = [
    "Software Developer",
    "Data Analyst",
    "UI/UX Designer",
    "Cybersecurity Analyst",
    "Game Developer",
]


def create_dataset():
    """Create synthetic examples for learning and testing."""

    rows = []

    # Each profile contains ratings from 1 to 5.
    # The label is generated from transparent prototype rules.
    for programming in range(1, 6):
        for mathematics in range(1, 6):
            for creativity in range(1, 6):
                for communication in range(1, 6):
                    for design in range(1, 6):
                        for security in range(1, 6):
                            for gaming in range(1, 6):
                                problem_solving = round(
                                    (programming + mathematics) / 2
                                )

                                scores = {
                                    "Software Developer":
                                        programming * 2
                                        + problem_solving,

                                    "Data Analyst":
                                        mathematics * 2
                                        + problem_solving,

                                    "UI/UX Designer":
                                        design * 2
                                        + creativity * 2,

                                    "Cybersecurity Analyst":
                                        security * 2
                                        + problem_solving,

                                    "Game Developer":
                                        programming
                                        + gaming * 2
                                        + creativity,
                                }

                                career = max(
                                    scores,
                                    key=scores.get
                                )

                                rows.append({
                                    "programming": programming,
                                    "mathematics": mathematics,
                                    "creativity": creativity,
                                    "communication": communication,
                                    "problem_solving": problem_solving,
                                    "design": design,
                                    "security": security,
                                    "gaming": gaming,
                                    "career": career,
                                })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_FILE, index=False)
    return df


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing synthetic career dataset...")
    df = create_dataset()

    X = df[FEATURES]
    y = df["career"]

    print(f"Total examples: {len(df)}")
    print("\nExamples per career:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n--- Evaluation on held-out synthetic data ---")
    print(f"Training examples: {len(X_train)}")
    print(f"Testing examples: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=CAREERS,
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=CAREERS),
        index=CAREERS,
        columns=CAREERS,
    ))

    joblib.dump({
        "model": model,
        "features": FEATURES,
        "careers": CAREERS,
        "dataset_type": "synthetic_rule_generated",
    }, MODEL_FILE)

    print(f"\nDataset saved to: {DATA_FILE}")
    print(f"Model saved to: {MODEL_FILE}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
