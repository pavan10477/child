"""WHO Child Growth Standards screening calculations for children under five."""

import csv
import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path


REFERENCE_DIR = Path(__file__).resolve().parent / "data" / "who_growth_standards"
WHO_DAYS_PER_MONTH = 30.4375
LAST_REFERENCE_AGE_DAYS = 1826
LMS = tuple[float, float, float]


@dataclass(frozen=True)
class GrowthStandards:
    length_height_for_age: dict[tuple[int, int], LMS]
    weight_for_age: dict[tuple[int, int], LMS]
    weight_for_length_height: dict[tuple[int, int, str], LMS]


def _load_tsv(
    filename: str, index_fields: tuple[str, ...], measure_field: str | None = None
) -> dict[tuple[object, ...], LMS]:
    path = REFERENCE_DIR / filename
    with path.open(encoding="utf-8", newline="") as reference_file:
        rows = csv.DictReader(reference_file, delimiter="\t")
        expected = set(index_fields) | {"l", "m", "s"}
        if rows.fieldnames is None or not expected.issubset(rows.fieldnames):
            raise RuntimeError(f"WHO reference file has an unexpected format: {path}")

        table: dict[tuple[object, ...], LMS] = {}
        for row in rows:
            key_values: list[object] = []
            for field in index_fields:
                value = row[field]
                if field == "sex":
                    key_values.append(int(value))
                elif field == "age":
                    key_values.append(int(value))
                elif field == measure_field:
                    key_values.append(int(round(float(value) * 10)))
                else:
                    key_values.append(value)
            key = tuple(key_values)
            if key in table:
                raise RuntimeError(f"Duplicate WHO reference entry in {path}: {key}")
            table[key] = (float(row["l"]), float(row["m"]), float(row["s"]))

    if not table:
        raise RuntimeError(f"WHO reference file contains no rows: {path}")
    return table


@lru_cache(maxsize=1)
def load_growth_standards() -> GrowthStandards:
    length_height_for_age = _load_tsv("lenanthro.txt", ("sex", "age"))
    weight_for_age = _load_tsv("weianthro.txt", ("sex", "age"))

    weight_for_length_height = _load_tsv(
        "wflanthro.txt", ("sex", "length", "lorh"), measure_field="length"
    )
    weight_for_length_height.update(
        _load_tsv(
            "wfhanthro.txt", ("sex", "height", "lorh"), measure_field="height"
        )
    )
    return GrowthStandards(
        length_height_for_age=length_height_for_age,
        weight_for_age=weight_for_age,
        weight_for_length_height=weight_for_length_height,
    )


def _parse_positive_decimal(value: str, label: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError):
        raise ValueError(f"Enter a valid {label}.") from None
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{label.capitalize()} must be greater than zero.")
    try:
        if not math.isfinite(float(parsed)):
            raise ValueError(f"Enter a finite {label} value.")
    except OverflowError:
        raise ValueError(f"Enter a finite {label} value.") from None
    return parsed


def _compute_zscore(value: float, lms: LMS, adjust_extremes: bool) -> float:
    l, m, s = lms
    if l == 0:
        z_score = math.log(value / m) / s
    else:
        z_score = ((value / m) ** l - 1) / (l * s)

    if not adjust_extremes or -3 <= z_score <= 3:
        return round(z_score, 2)

    def value_at_standard_deviations(standard_deviations: int) -> float:
        if l == 0:
            return m * math.exp(s * standard_deviations)
        return m * (1 + l * s * standard_deviations) ** (1 / l)

    if z_score > 3:
        sd3 = value_at_standard_deviations(3)
        sd2 = value_at_standard_deviations(2)
        z_score = 3 + (value - sd3) / (sd3 - sd2)
    else:
        sd3 = value_at_standard_deviations(-3)
        sd2 = value_at_standard_deviations(-2)
        z_score = -3 + (value - sd3) / (sd2 - sd3)
    return round(z_score, 2)


def _weight_for_length_height_lms(
    standards: GrowthStandards, sex: int, mode: str, measurement_cm: Decimal
) -> LMS | None:
    scaled_value = measurement_cm * 10
    lower_tick = int(scaled_value)
    fraction = float(scaled_value - lower_tick)
    lower = standards.weight_for_length_height.get((sex, lower_tick, mode))
    if lower is None:
        return None
    if fraction == 0:
        return lower

    upper = standards.weight_for_length_height.get((sex, lower_tick + 1, mode))
    if upper is None:
        return None
    interpolated = tuple(
        low + fraction * (high - low) for low, high in zip(lower, upper)
    )
    return interpolated[0], interpolated[1], interpolated[2]


def _indicator(name: str, z_score: float, status: str) -> dict[str, str]:
    return {"name": name, "z_score": f"{z_score:.2f}", "status": status}


def _height_status(z_score: float) -> str:
    if z_score < -3:
        return "Below -3 SD (severe stunting threshold)"
    if z_score < -2:
        return "Below -2 SD (stunting threshold)"
    return "At or above -2 SD"


def _weight_status(z_score: float) -> str:
    if z_score < -3:
        return "Below -3 SD (severe underweight threshold)"
    if z_score < -2:
        return "Below -2 SD (underweight threshold)"
    return "At or above -2 SD"


def _weight_for_length_height_status(z_score: float) -> str:
    if z_score < -3:
        return "Below -3 SD (severe wasting threshold)"
    if z_score < -2:
        return "Below -2 SD (wasting threshold)"
    if z_score > 3:
        return "Above +3 SD"
    if z_score > 2:
        return "Above +2 SD (overweight threshold)"
    return "Between -2 SD and +2 SD"


def calculate_screening(
    *,
    birth_date: date,
    measurement_date: date,
    sex: str,
    weight_kg: str,
    length_height_cm: str,
    measurement_position: str,
    oedema: str,
) -> dict[str, object]:
    if measurement_date < birth_date:
        raise ValueError("Measurement date cannot be earlier than the birth date.")
    if measurement_date > date.today():
        raise ValueError("Measurement date cannot be in the future.")

    age_days = (measurement_date - birth_date).days
    if age_days > LAST_REFERENCE_AGE_DAYS or age_days / WHO_DAYS_PER_MONTH >= 60:
        raise ValueError(
            "WHO under-five screening is available here only for children younger than 60 months."
        )

    sex_code = {"Male": 1, "Female": 2}.get(sex)
    if sex_code is None:
        raise ValueError("Select the child's sex used by the WHO growth reference.")
    if measurement_position not in {"length", "height"}:
        raise ValueError("Select whether the measurement was taken lying down or standing.")
    if oedema not in {"no", "yes", "unsure"}:
        raise ValueError("Select an option for bilateral pitting oedema.")

    weight = _parse_positive_decimal(weight_kg, "weight")
    stature = _parse_positive_decimal(length_height_cm, "length or height")
    standards = load_growth_standards()
    indicators: list[dict[str, str]] = []

    adjustment = 0.7 if age_days < 731 and measurement_position == "height" else 0.0
    if age_days >= 731 and measurement_position == "length":
        adjustment = -0.7
    standardized_stature = stature + Decimal(str(adjustment))

    length_height_lms = standards.length_height_for_age.get((sex_code, age_days))
    if length_height_lms is None:
        raise RuntimeError(f"No WHO length/height-for-age reference for day {age_days}.")
    height_z = _compute_zscore(float(standardized_stature), length_height_lms, False)
    indicators.append(_indicator("Length/height-for-age", height_z, _height_status(height_z)))

    if oedema == "no":
        weight_age_lms = standards.weight_for_age.get((sex_code, age_days))
        if weight_age_lms is None:
            raise RuntimeError(f"No WHO weight-for-age reference for day {age_days}.")
        weight_z = _compute_zscore(float(weight), weight_age_lms, True)
        indicators.append(_indicator("Weight-for-age", weight_z, _weight_status(weight_z)))

        reference_mode = "L" if age_days < 731 else "H"
        weight_length_height_lms = _weight_for_length_height_lms(
            standards, sex_code, reference_mode, standardized_stature
        )
        if weight_length_height_lms is None:
            valid_range = "45–110 cm" if reference_mode == "L" else "65–120 cm"
            indicators.append(
                {
                    "name": "Weight-for-length/height",
                    "z_score": "Not calculated",
                    "status": f"WHO reference range for this measurement is {valid_range}.",
                }
            )
        else:
            weight_length_height_z = _compute_zscore(
                float(weight), weight_length_height_lms, True
            )
            indicators.append(
                _indicator(
                    "Weight-for-length/height",
                    weight_length_height_z,
                    _weight_for_length_height_status(weight_length_height_z),
                )
            )
    else:
        status = (
            "Not calculated because oedema was reported."
            if oedema == "yes"
            else "Not calculated because oedema status is unknown."
        )
        for name in ("Weight-for-age", "Weight-for-length/height"):
            indicators.append(
                {"name": name, "z_score": "Not calculated", "status": status}
            )

    return {
        "age_days": age_days,
        "indicators": indicators,
        "position_adjustment_cm": f"{adjustment:+.1f}",
        "standardized_measurement_cm": f"{standardized_stature:.1f}",
    }
