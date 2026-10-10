from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
import json
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import joblib
import pandas as pd
from datetime import datetime

# Load the trained ML model once when Flask starts
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "ml"
    / "models"
    / "career_model.joblib"
)

career_model = joblib.load(MODEL_PATH)
MODEL_FEATURES = career_model["features"]
MODEL_CAREERS = career_model["careers"]

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "careercompass.db"
app = Flask(__name__)

def predict_careers_ml(profile):
    """
    profile should contain ratings from 1 to 5
    for the model's expected features.
    """

    profile = profile.copy()

    # Calculate problem-solving if the questionnaire
    # doesn't already provide it.
    if "problem_solving" not in profile:
        programming = float(profile.get("programming", 3))
        mathematics = float(profile.get("mathematics", 3))

        profile["problem_solving"] = round(
            (programming + mathematics) / 2
        )

    # Prepare features in exactly the training order
    values = []

    for feature in MODEL_FEATURES:
        value = float(profile.get(feature, 3))

        if not 1 <= value <= 5:
            raise ValueError(
                f"{feature} must be between 1 and 5"
            )

        values.append(value)

    input_data = pd.DataFrame(
        [values],
        columns=MODEL_FEATURES
    )

    probabilities = career_model["model"].predict_proba(
        input_data
    )[0]

    classes = career_model["model"].classes_

    results = [
        {
            "career": str(career),
            "score": round(float(probability) * 100, 1)
        }
        for career, probability in zip(classes, probabilities)
    ]

    results.sort(key=lambda item: item["score"], reverse=True)

    return results


app.secret_key = "change-this-development-key"

# Beginner-friendly, explainable rule-based career profiles.
# Values represent how relevant each dimension is to a career (0-5).
DIMENSIONS = {
    "programming": "Programming",
    "logic": "Logical problem-solving",
    "creativity": "Creativity",
    "math": "Mathematics",
    "data": "Data analysis",
    "security": "Cybersecurity",
    "design": "Visual/UI design",
    "communication": "Communication",
    "business": "Business and product thinking",
    "hardware": "Hardware and systems",
}

CAREERS = [
    {
        "id": "software_developer", "name": "Software Developer",
        "summary": "Build applications, tools and software used by people and organizations.",
        "weights": {"programming": 5, "logic": 5, "creativity": 3, "math": 3, "data": 2, "security": 1, "design": 2, "communication": 3, "business": 2, "hardware": 1},
        "skills": ["Python or C++", "Data structures and algorithms", "Git and GitHub", "Object-oriented programming", "Testing and debugging"],
        "projects": ["To-do list web app", "Student result manager", "Personal portfolio website"],
        "roadmap": ["Learn one programming language and its fundamentals", "Practice functions, collections and object-oriented programming", "Learn Git and basic data structures", "Build two small projects and publish them on GitHub"]
    },
    {
        "id": "game_developer", "name": "Game Developer",
        "summary": "Create gameplay systems, mechanics, tools and interactive worlds.",
        "weights": {"programming": 4, "logic": 4, "creativity": 5, "math": 3, "data": 1, "security": 0, "design": 4, "communication": 2, "business": 2, "hardware": 2},
        "skills": ["C# or C++", "Godot or Unity", "Game physics basics", "Version control with Git", "Level and gameplay design"],
        "projects": ["2D platformer", "Puzzle game", "Small top-down survival game"],
        "roadmap": ["Choose Godot or Unity and learn its editor", "Learn scenes, nodes or game objects and input", "Build a small playable prototype", "Add menus, sound, polish and publish a demo"]
    },
    {
        "id": "web_developer", "name": "Web Developer",
        "summary": "Build websites and web applications, from interactive pages to server-backed products.",
        "weights": {"programming": 4, "logic": 4, "creativity": 3, "math": 1, "data": 2, "security": 2, "design": 4, "communication": 3, "business": 3, "hardware": 0},
        "skills": ["HTML", "CSS and responsive design", "JavaScript", "HTTP and APIs", "A backend framework and SQL"],
        "projects": ["Responsive portfolio", "Campus complaint tracker", "Study planner web app"],
        "roadmap": ["Learn semantic HTML and CSS layouts", "Build responsive pages", "Learn JavaScript and browser events", "Connect a frontend to Flask and SQLite"]
    },
    {
        "id": "ai_ml_engineer", "name": "AI / Machine Learning Engineer",
        "summary": "Build systems that learn patterns from data and support predictions or automation.",
        "weights": {"programming": 4, "logic": 4, "creativity": 3, "math": 5, "data": 5, "security": 1, "design": 1, "communication": 2, "business": 2, "hardware": 1},
        "skills": ["Python", "Probability and statistics", "Linear algebra basics", "Pandas and data cleaning", "Model evaluation with scikit-learn"],
        "projects": ["Student performance data explorer", "Spam message classifier", "Movie recommendation prototype"],
        "roadmap": ["Learn Python and basic data handling", "Study probability, statistics and essential linear algebra", "Practice with Pandas and visualizations", "Train and evaluate a small scikit-learn model"]
    },
    {
        "id": "data_analyst", "name": "Data Analyst",
        "summary": "Turn raw data into useful findings, reports and decisions.",
        "weights": {"programming": 3, "logic": 4, "creativity": 2, "math": 4, "data": 5, "security": 1, "design": 3, "communication": 4, "business": 4, "hardware": 0},
        "skills": ["Excel or spreadsheets", "SQL", "Statistics", "Python with Pandas", "Charts and data storytelling"],
        "projects": ["College survey dashboard", "Public dataset analysis", "Student attendance trends report"],
        "roadmap": ["Learn spreadsheets and data cleaning", "Learn SQL queries and joins", "Practice descriptive statistics", "Create a dashboard and explain the findings clearly"]
    },
    {
        "id": "cybersecurity_analyst", "name": "Cybersecurity Analyst",
        "summary": "Help protect systems, investigate risks and improve digital security.",
        "weights": {"programming": 3, "logic": 5, "creativity": 2, "math": 2, "data": 2, "security": 5, "design": 0, "communication": 3, "business": 2, "hardware": 3},
        "skills": ["Networking fundamentals", "Linux basics", "Security concepts", "Python scripting", "Log analysis and safe lab practice"],
        "projects": ["Password-strength checker", "Local log-file analyzer", "Home-lab network diagram"],
        "roadmap": ["Learn operating-system and networking basics", "Practice Linux commands in a safe environment", "Study common security principles", "Build defensive tools and document findings responsibly"]
    },
    {
        "id": "ui_ux_designer", "name": "UI/UX Designer",
        "summary": "Research user needs and design clear, accessible digital experiences.",
        "weights": {"programming": 1, "logic": 3, "creativity": 5, "math": 0, "data": 2, "security": 0, "design": 5, "communication": 5, "business": 3, "hardware": 0},
        "skills": ["User research", "Wireframing", "Prototyping", "Visual hierarchy and accessibility", "Usability testing"],
        "projects": ["Redesign a college website page", "Student app prototype", "Usability test report"],
        "roadmap": ["Study design principles and accessibility", "Learn wireframes and clickable prototypes", "Interview users about a real problem", "Test a design with users and iterate"]
    },
    {
        "id": "devops_cloud", "name": "Cloud / DevOps Engineer",
        "summary": "Help applications run reliably through automation, deployment and infrastructure.",
        "weights": {"programming": 3, "logic": 4, "creativity": 2, "math": 1, "data": 2, "security": 4, "design": 0, "communication": 3, "business": 2, "hardware": 4},
        "skills": ["Linux and command line", "Git", "Networking", "Docker basics", "Cloud and CI/CD fundamentals"],
        "projects": ["Deploy a Flask app", "Automated backup script", "Simple CI workflow"],
        "roadmap": ["Learn Linux, networking and Git", "Deploy a small application", "Learn containers and environment variables", "Automate tests and deployment in a practice project"]
    },
]


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            education TEXT NOT NULL,
            answers_json TEXT NOT NULL,
            results_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")



def recommend(answers):
    """
    Hybrid career recommender:
    70% weighted skill/interest score
    30% cosine similarity using scikit-learn
    """

    # Convert the student's answers into a numerical profile.
    student_vector = [
        int(answers.get(dimension, 3)) / 5.0
        for dimension in DIMENSIONS
    ]

    # Build a numerical profile for every career.
    career_vectors = [
        [
            career["weights"].get(dimension, 0) / 5.0
            for dimension in DIMENSIONS
        ]
        for career in CAREERS
    ]

    # Compare the student profile with each career profile.
    similarities = cosine_similarity(
        [student_vector],
        career_vectors
    )[0]

    ranked = []

    for index, career in enumerate(CAREERS):
        weighted_sum = 0.0
        max_possible = 0.0

        for dimension in DIMENSIONS:
            relevance = career["weights"].get(dimension, 0)
            answer = int(answers.get(dimension, 3))

            weighted_sum += (answer / 5.0) * relevance
            max_possible += relevance

        # Original explainable recommendation score.
        rule_score = (
            weighted_sum / max_possible * 100
            if max_possible else 0
        )

        # Similarity between 0 and 1, converted to a percentage.
        similarity_score = float(similarities[index]) * 100

        # Combine the two scoring methods.
        final_score = round(
            0.70 * rule_score + 0.30 * similarity_score
        )

        # Identify areas the student may want to develop.
        gaps = []

        for dimension, label in DIMENSIONS.items():
            relevance = career["weights"].get(dimension, 0)
            answer = int(answers.get(dimension, 3))

            if relevance >= 3 and answer <= 2:
                gaps.append({
                    "dimension": label,
                    "suggestion": (
                        f"Build confidence in {label.lower()} "
                        "through beginner exercises."
                    )
                })

        ranked.append({
            "id": career["id"],
            "name": career["name"],
            "summary": career["summary"],
            "score": final_score,
            "skills": career["skills"],
            "projects": career["projects"],
            "roadmap": career["roadmap"],
            "gaps": gaps[:4]
        })

    # Return the three highest-scoring career matches.
    ranked.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return ranked[:3]



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/assessment", methods=["GET", "POST"])
def assessment():
    if request.method == "POST":
        name = request.form.get("student_name", "").strip()[:80]
        education = request.form.get("education", "B.Tech 1st year").strip()[:100]
        if not name:
            flash("Please enter your name to continue.", "error")
            return redirect(url_for("assessment"))
        answers = {}
        try:
            for dimension in DIMENSIONS:
                value = int(request.form.get(dimension, "3"))
                if value < 1 or value > 5:
                    raise ValueError
                answers[dimension] = value
        except ValueError:
            flash("Please answer every rating question using a value from 1 to 5.", "error")
            return redirect(url_for("assessment"))
        # Get recommendations from the existing engine
        results = recommend(answers)

        # Convert questionnaire answers into ML model features
        ml_profile = {
            "programming": answers.get("programming", 3),
            "mathematics": answers.get("math", 3),
            "creativity": answers.get("creativity", 3),
            "communication": answers.get("communication", 3),
            "design": answers.get("design", 3),
            "security": answers.get("security", 3),
            "gaming": 3,
            "problem_solving": round(
                (
                    answers.get("programming", 3)
                    + answers.get("logic", 3)
                ) / 2
            ),
        }

        # Get machine-learning predictions
        ml_results = predict_careers_ml(ml_profile)

        # Match ML predictions to the existing career results
        ml_scores = {
            item["career"]: item["score"]
            for item in ml_results
        }

        # Combine rule-based and ML scores
        for result in results:
            ml_score = ml_scores.get(result["name"])

            if ml_score is not None:
                result["ml_score"] = ml_score
                result["score"] = round(
                    0.70 * result["score"]
                    + 0.30 * ml_score
                )
            else:
                result["ml_score"] = None

        # Reorder results using the combined score
        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO assessments (student_name, education, answers_json, results_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (name, education, json.dumps(answers), json.dumps(results), datetime.now().isoformat(timespec="seconds"))
            )
            assessment_id = cursor.lastrowid
        return redirect(url_for("results", assessment_id=assessment_id))
    return render_template("assessment.html", dimensions=DIMENSIONS)


@app.route("/results/<int:assessment_id>")
def results(assessment_id):
    with get_db() as conn:
        record = conn.execute("SELECT * FROM assessments WHERE id = ?", (assessment_id,)).fetchone()
    if record is None:
        flash("That assessment could not be found.", "error")
        return redirect(url_for("home"))
    return render_template(
        "results.html", record=record,
        results=json.loads(record["results_json"]),
        answers=json.loads(record["answers_json"]), dimensions=DIMENSIONS
    )


@app.route("/history")
def history():
    with get_db() as conn:
        records = conn.execute("SELECT id, student_name, education, created_at FROM assessments ORDER BY id DESC LIMIT 50").fetchall()
    return render_template("history.html", records=records)


@app.route("/health")
def health():
    return {"status": "ok", "application": "CareerCompass AI", "recommendation_engine": "hybrid-rule-cosine-ml-v2"}


if __name__ == "__main__":
    init_db()
    app.run(debug=True)



