from flask import Flask, render_template, request, redirect, url_for, session, flash
from dotenv import load_dotenv
import os

from db import init_db, create_user, verify_user, save_prediction, get_predictions_for_user
from utils.ml_model import (
    predict_energy_cost,
    get_available_building_types,
    get_renewable_advice,
)
from utils.weather_api import get_weather_forecast
from utils.twilio_api import send_energy_tip_sms

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

init_db()  # creates tables on first run, no-ops if they already exist


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password:
            return render_template("signup.html", error="Username and password are required.")

        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match.")

        if len(password) < 6:
            return render_template("signup.html", error="Password must be at least 6 characters.")

        ok, result = create_user(username, password)
        if not ok:
            return render_template("signup.html", error=result)  # result is the error message

        # Auto-login after successful signup
        session["user"] = username
        session["user_id"] = result  # result is the new user's id on success
        return redirect(url_for("predictions"))

    return render_template("signup.html")


@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")

    user_id = verify_user(username, password)
    if user_id is not None:
        session["user"] = username
        session["user_id"] = user_id
        return redirect(url_for("predictions"))

    return render_template("home.html", error="Invalid username or password")


@app.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("user_id", None)
    session.pop("last_result", None)
    return redirect(url_for("home"))


@app.route("/predictions", methods=["GET", "POST"])
def predictions():
    if "user" not in session:
        return redirect(url_for("home"))

    result = None
    weather = None
    building_types = get_available_building_types()

    if request.method == "POST":
        building_type = request.form.get("building_type")
        sqft = request.form.get("sqft")
        zip_code = request.form.get("zip_code")
        country = request.form.get("country")

        try:
            sqft_value = float(sqft)
        except (TypeError, ValueError):
            sqft_value = 0.0

        result = predict_energy_cost(
            square_feet=sqft_value,
            building_type=building_type,
        )

        # NOTE: estimated_savings is NOT a model output — the real ML model
        # only predicts usage/cost. This is a simple heuristic (assumes a
        # 12% efficiency improvement is achievable) shown to give users a
        # rough sense of potential savings, not a data-driven prediction.
        result["estimated_savings"] = round(result["estimated_cost"] * 0.12, 2)

        weather = get_weather_forecast(zip_code, country)

        # Persist this prediction to the user's history
        save_prediction(
            user_id=session["user_id"],
            building_type=result["building_type_used"],
            square_feet=sqft_value,
            zip_code=zip_code,
            country=country,
            predicted_usage_kwh=result["predicted_usage_kwh"],
            estimated_cost=result["estimated_cost"],
            estimated_savings=result["estimated_savings"],
        )

        # Stash the latest result in session so the SMS route can reuse it
        # without the user having to resubmit the form.
        session["last_result"] = result

    return render_template(
        "predictions.html",
        result=result,
        weather=weather,
        building_types=building_types,
    )


@app.route("/send-tip-sms", methods=["POST"])
def send_tip_sms():
    if "user" not in session:
        return redirect(url_for("home"))

    phone_number = request.form.get("phone_number")
    result = session.get("last_result")

    if not result:
        flash("Run a prediction first, then text yourself the result.", "error")
        return redirect(url_for("predictions"))

    message = (
        f"EcoGrid tip: Your estimated usage is {result['predicted_usage_kwh']} kWh "
        f"(${result['estimated_cost']}) for a {result['building_type_used']} building. "
        f"Check the app for energy-saving tips!"
    )

    sms_result = send_energy_tip_sms(phone_number, message)

    if sms_result["success"]:
        flash("Text sent! Check your phone.", "success")
    else:
        flash(f"Couldn't send text: {sms_result['error']}", "error")

    return redirect(url_for("predictions"))


@app.route("/renewable-advisor")
def renewable_advisor():
    if "user" not in session:
        return redirect(url_for("home"))

    last_result = session.get("last_result")

    if last_result:
        # Tie the advisor to the user's most recent real prediction:
        # predicted_usage_kwh is hourly (see ml_model.py docstring), so we
        # scale to a full day's usage for the 24-hour demand curve.
        daily_usage_kwh = last_result["predicted_usage_kwh"] * 24
        building_type = last_result["building_type_used"]
    else:
        # No prediction yet this session — fall back to a reasonable
        # default rather than crashing, but flag it so it's not confused
        # with a real personalized forecast.
        flash(
            "Run a prediction first for a personalized renewable schedule. "
            "Showing a default Office estimate for now.",
            "error",
        )
        daily_usage_kwh = 450
        building_type = "Office"

    advice = get_renewable_advice(usage_kwh=daily_usage_kwh, building_type=building_type)
    return render_template("renewable_advisor.html", advice=advice)


@app.route("/analytics")
def analytics():
    if "user" not in session:
        return redirect(url_for("home"))

    history = get_predictions_for_user(session["user_id"], limit=30)

    if history:
        avg_usage = round(
            sum(d["predicted_usage_kwh"] for d in history) / len(history), 1
        )
        total_cost = round(sum(d["estimated_cost"] for d in history), 2)
    else:
        avg_usage = 0
        total_cost = 0

    return render_template(
        "analytics.html",
        history=history,
        avg_usage=avg_usage,
        total_cost=total_cost,
    )


@app.route("/resources")
def resources():
    return render_template("resources.html")


if __name__ == "__main__":
    app.run(debug=True)