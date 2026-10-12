from contextlib import closing
from functools import wraps
import hmac
import os
import secrets
import sqlite3
from datetime import date
from pathlib import Path
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from risk_classifier import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RiskClassifier,
    parse_features,
)
from who_screening import calculate_screening

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config["DATABASE"] = os.path.join(app.instance_path, "users.sqlite3")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RISK_MODEL_PATH = os.path.join(BASE_DIR, "nutritional_risk_model.json")
RISK_EXAMPLES_PATH = os.path.join(
    BASE_DIR, "data", "nutritional_risk_examples_unverified.csv"
)

def init_db():
    os.makedirs(os.path.dirname(app.config["DATABASE"]), exist_ok=True)
    with closing(sqlite3.connect(app.config["DATABASE"])) as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            )
            """
        )
        db.commit()


def csrf_token():
    token = session.get("csrf_token")
    if token is None:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def valid_csrf_token():
    submitted = request.form.get("csrf_token", "")
    expected = session.get("csrf_token", "")
    return bool(expected and submitted and hmac.compare_digest(submitted, expected))


def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if "user_id" not in session:
            flash("Please sign in to continue.", "error")
            return redirect(url_for("login"))
        return view(**kwargs)

    return wrapped_view


@app.context_processor
def inject_csrf_token():
    return {"csrf_token": csrf_token}


@app.after_request
def set_session_cookie_options(response):
    response.headers["Cache-Control"] = "no-store"
    return response


init_db()


def load_risk_classifier():
    if os.path.isfile(RISK_MODEL_PATH):
        return RiskClassifier.from_json(Path(RISK_MODEL_PATH).resolve()), False
    return RiskClassifier.from_csv(Path(RISK_EXAMPLES_PATH)), True


@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        if not valid_csrf_token():
            flash("Your form expired. Please try again.", "error")
            return render_template("login.html")

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with closing(sqlite3.connect(app.config["DATABASE"])) as db:
            db.row_factory = sqlite3.Row
            user = db.execute(
                "SELECT id, email, password_hash FROM users WHERE email = ?",
                (email,),
            ).fetchone()

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        session.clear()
        session["user_id"] = user["id"]
        session["user_email"] = user["email"]
        flash("Signed in successfully.", "success")
        return redirect(url_for("index"))

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        if not valid_csrf_token():
            flash("Your form expired. Please try again.", "error")
            return render_template("signup.html")

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
            flash("Enter a valid email address.", "error")
            return render_template("signup.html")
        if len(password) < 8:
            flash("Password must be at least 8 characters long.", "error")
            return render_template("signup.html")
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")

        try:
            with closing(sqlite3.connect(app.config["DATABASE"])) as db:
                db.execute(
                    "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                    (email, generate_password_hash(password)),
                )
                db.commit()
        except sqlite3.IntegrityError:
            flash("An account with that email already exists. Please sign in.", "error")
            return render_template("signup.html")

        flash("Account created. Please sign in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    if not valid_csrf_token():
        flash("Your form expired. Please try again.", "error")
        return redirect(url_for("index"))
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("login"))


@app.route("/demo", methods=["GET", "POST"])
@login_required
def index():
    result = None
    error = None
    classifier, demo_mode = load_risk_classifier()

    if request.method == "POST":
        if not valid_csrf_token():
            error = "Your form expired. Refresh the page and try again."
        else:
            try:
                birth_date = date.fromisoformat(request.form.get("birth_date", ""))
                measurement_date = date.fromisoformat(
                    request.form.get("measurement_date", "")
                )
                age_days = (measurement_date - birth_date).days
                if age_days < 0:
                    raise ValueError("Measurement date cannot be earlier than the birth date.")
                if measurement_date > date.today():
                    raise ValueError("Measurement date cannot be in the future.")

                row = {
                    field: request.form.get(field, "")
                    for field in CATEGORICAL_FEATURES
                }
                row.update(
                    {
                        field: request.form.get(field, "")
                        for field in NUMERIC_FEATURES
                        if field != "Age"
                    }
                )
                row["Age"] = str(age_days / 365.25)
                sample = parse_features(row)
                prediction, votes, neighbor_count = classifier.predict(sample)
                screening = None
                screening_note = None
                if age_days <= 1826 and age_days / 30.4375 < 60:
                    screening = calculate_screening(
                        birth_date=birth_date,
                        measurement_date=measurement_date,
                        sex=request.form.get("Gender", ""),
                        weight_kg=request.form.get("Weight_kg", ""),
                        length_height_cm=request.form.get("Height_cm", ""),
                        measurement_position=request.form.get(
                            "measurement_position", ""
                        ),
                        oedema=request.form.get("oedema", ""),
                    )
                else:
                    screening_note = (
                        "WHO under-five growth indicators are available only for "
                        "children younger than 60 months."
                    )

                reported_factors = []
                if request.form.get("Fruits_Veg", "").casefold() == "low":
                    reported_factors.append("Low fruit and vegetable intake selected")
                if request.form.get("Protein", "").casefold() == "low":
                    reported_factors.append("Low protein intake selected")
                if request.form.get("Physical_Activity", "").casefold() == "low":
                    reported_factors.append("Low physical activity selected")
                if request.form.get("Previous_Illness", "").casefold() == "yes":
                    reported_factors.append("Previous illness reported")
                if request.form.get("Sanitation", "").casefold() == "poor":
                    reported_factors.append("Poor sanitation selected")

                result = {
                    "label": prediction,
                    "votes": votes,
                    "neighbor_count": neighbor_count,
                    "demo_mode": demo_mode,
                    "age_years": f"{age_days / 365.25:.1f}",
                    "gender": request.form.get("Gender", ""),
                    "height_cm": request.form.get("Height_cm", ""),
                    "weight_kg": request.form.get("Weight_kg", ""),
                    "meals_day": request.form.get("Meals_Day", ""),
                    "fruits_veg": request.form.get("Fruits_Veg", ""),
                    "protein": request.form.get("Protein", ""),
                    "physical_activity": request.form.get("Physical_Activity", ""),
                    "previous_illness": request.form.get("Previous_Illness", ""),
                    "sanitation": request.form.get("Sanitation", ""),
                    "screening": screening,
                    "screening_note": screening_note,
                    "reported_factors": reported_factors,
                }
            except ValueError as exc:
                error = str(exc)

    return render_template(
        "index.html",
        result=result,
        error=error,
        demo_mode=demo_mode,
        today=date.today().isoformat(),
    )


@app.route("/who-screening", methods=["GET", "POST"])
@login_required
def who_screening():
    result = None
    error = None
    today = date.today().isoformat()

    if request.method == "POST":
        if not valid_csrf_token():
            error = "Your form expired. Refresh the page and try again."
        else:
            try:
                birth_date = date.fromisoformat(request.form.get("birth_date", ""))
                measurement_date = date.fromisoformat(
                    request.form.get("measurement_date", "")
                )
                result = calculate_screening(
                    birth_date=birth_date,
                    measurement_date=measurement_date,
                    sex=request.form.get("sex", ""),
                    weight_kg=request.form.get("weight_kg", ""),
                    length_height_cm=request.form.get("length_height_cm", ""),
                    measurement_position=request.form.get(
                        "measurement_position", ""
                    ),
                    oedema=request.form.get("oedema", ""),
                )
            except ValueError as exc:
                error = str(exc)

    return render_template(
        "who_screening.html",
        result=result,
        error=error,
        today=today,
    )

if __name__ == "__main__":
    app.run(debug=True)
