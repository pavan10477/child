"""Dependency-free nearest-neighbor classifier for a supplied risk dataset."""

import csv
import json
import math
from pathlib import Path
from typing import Mapping


NUMERIC_FEATURES = ("Age", "Height_cm", "Weight_kg", "Meals_Day")
CATEGORICAL_FEATURES = (
    "Gender",
    "Fruits_Veg",
    "Protein",
    "Physical_Activity",
    "Previous_Illness",
    "Sanitation",
)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Nutritional_Risk"
CSV_COLUMNS = FEATURES + (TARGET,)
SUPPORTED_LABELS = {"Low", "Moderate", "High"}
NEIGHBOR_COUNT = 5


def parse_features(row: Mapping[str, str]) -> dict[str, object]:
    record: dict[str, object] = {}
    for field in NUMERIC_FEATURES:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Invalid numeric value for {field}.") from None
        if not math.isfinite(value):
            raise ValueError(f"{field} must be a finite number.")
        if value < 0 or (field in {"Height_cm", "Weight_kg"} and value == 0):
            raise ValueError(f"{field} must be greater than zero.")
        record[field] = value

    for field in CATEGORICAL_FEATURES:
        value = row.get(field, "").strip().casefold()
        if not value:
            raise ValueError(f"{field} cannot be blank.")
        record[field] = value

    return record


def parse_record(row: Mapping[str, str]) -> dict[str, object]:
    record = parse_features(row)
    label = row.get(TARGET, "").strip().title()
    if label not in SUPPORTED_LABELS:
        raise ValueError(f"{TARGET} must be Low, Moderate, or High.")
    record[TARGET] = label
    return record


class RiskClassifier:
    def __init__(self, records: list[dict[str, object]]) -> None:
        if not records:
            raise ValueError("The training dataset contains no records.")
        self.records = records
        self.ranges = {
            field: max(float(row[field]) for row in records)
            - min(float(row[field]) for row in records)
            for field in NUMERIC_FEATURES
        }

    @classmethod
    def from_csv(cls, csv_path: Path) -> "RiskClassifier":
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Verified training data not found: {csv_path}. "
                "Add the verified full dataset using the documented CSV columns."
            )

        with csv_path.open(encoding="utf-8-sig", newline="") as data_file:
            reader = csv.DictReader(data_file)
            if reader.fieldnames is None or not set(CSV_COLUMNS).issubset(reader.fieldnames):
                raise ValueError(
                    "Training CSV is missing required columns: " + ", ".join(CSV_COLUMNS)
                )
            records = []
            for line_number, row in enumerate(reader, start=2):
                try:
                    records.append(parse_record(row))
                except ValueError as exc:
                    raise ValueError(f"Training CSV line {line_number}: {exc}") from exc

        labels = {str(row[TARGET]) for row in records}
        missing_labels = SUPPORTED_LABELS - labels
        if missing_labels:
            raise ValueError(
                "Training data must include all three risk labels: "
                + ", ".join(sorted(missing_labels))
                + " are missing."
            )
        return cls(records)

    def predict(self, sample: Mapping[str, object]) -> tuple[str, int, int]:
        neighbors = sorted(
            (
                self._distance(sample, record),
                str(record[TARGET]),
                index,
            )
            for index, record in enumerate(self.records)
        )[: min(NEIGHBOR_COUNT, len(self.records))]

        votes: dict[str, int] = {}
        for _, label, _ in neighbors:
            votes[label] = votes.get(label, 0) + 1
        nearest_label = neighbors[0][1]
        prediction = max(
            votes,
            key=lambda label: (votes[label], label == nearest_label, label),
        )
        return prediction, votes[prediction], len(neighbors)

    def to_json(self, output_path: Path) -> None:
        payload = {"records": self.records}
        with output_path.open("w", encoding="utf-8") as model_file:
            json.dump(payload, model_file, allow_nan=False)

    @classmethod
    def from_json(cls, model_path: Path) -> "RiskClassifier":
        with model_path.open(encoding="utf-8") as model_file:
            payload = json.load(model_file)
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("The saved classifier contains no training records.")
        validated_records = [parse_record(record) for record in records]
        return cls(validated_records)

    def _distance(
        self, sample: Mapping[str, object], record: Mapping[str, object]
    ) -> float:
        distance = 0.0
        for field in NUMERIC_FEATURES:
            difference = float(sample[field]) - float(record[field])
            field_range = self.ranges[field]
            if field_range:
                difference /= field_range
            distance += difference * difference

        for field in CATEGORICAL_FEATURES:
            if sample[field] != record[field]:
                distance += 1.0
        return distance
