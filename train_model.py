"""Build a nearest-neighbor classifier from a verified nutritional-risk CSV."""

from pathlib import Path

from risk_classifier import RiskClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "nutritional_risk_data.csv"
MODEL_PATH = BASE_DIR / "nutritional_risk_model.json"


def main() -> None:
    classifier = RiskClassifier.from_csv(DATA_PATH)
    classifier.to_json(MODEL_PATH)
    print(f"Loaded {len(classifier.records)} dataset records.")
    print(f"Saved classifier to {MODEL_PATH}")
    print(
        "Training completed; this does not validate the data or establish "
        "medical accuracy. Evaluate with independent, representative data "
        "before interpreting any predictions."
    )


if __name__ == "__main__":
    main()
