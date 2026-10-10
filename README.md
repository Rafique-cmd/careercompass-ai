# CareerCompass AI — Beginner Project

A local web app for exploratory student career recommendations. Version 1 uses an explainable rule-based scoring engine; it does not require an AI API or machine-learning library.

## Requirements
- Python 3.10 or newer
- A web browser
- Internet is not required to run the app after Flask is installed. The optional Google Fonts import falls back to system fonts if offline.

## Run on Windows
1. Extract the ZIP file.
2. Open the `careercompass_ai` folder in VS Code.
3. Open the VS Code terminal (**Terminal → New Terminal**).
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
5. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, open Command Prompt in VS Code and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
6. Install Flask:
   ```powershell
   python -m pip install -r requirements.txt
   ```
7. Start the app:
   ```powershell
   python app.py
   ```
8. Open `http://127.0.0.1:5000` in your browser.
9. Stop the server with `Ctrl+C` in the terminal.

## Run on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000`.

## Project structure
```text
careercompass_ai/
├── app.py                 # Flask routes, career profiles, scoring and SQLite
├── requirements.txt       # Python dependencies
├── README.md              # Setup and project guide
├── .gitignore
├── careercompass.db       # Created automatically when the app runs
├── templates/
│   ├── base.html          # Shared navigation and layout
│   ├── index.html         # Landing page
│   ├── assessment.html    # Student assessment form
│   ├── results.html       # Top matches, skill gaps and roadmaps
│   └── history.html       # Saved assessments
└── static/
    ├── style.css          # Responsive styling
    └── app.js             # Slider values and form validation
```

## How recommendations are calculated
Each career has a relevance weight for ten dimensions. Each student rates dimensions from 1 to 5. The score is the weighted sum of the student's ratings divided by the maximum possible weighted score, then converted to a percentage. The app returns the top three careers.

These are **profile-match scores**, not probabilities of career success. The initial weights are illustrative and should be improved through research, student feedback and testing.

## Test checklist
- Open the home page and navigate to Assessment.
- Try submitting without a name; a validation message should appear.
- Complete the assessment and verify that three results appear.
- Open a roadmap for each career.
- Visit History and reopen a saved assessment.
- Restart the server and verify that saved assessments remain in SQLite.
- Visit `http://127.0.0.1:5000/health`; it should return a small JSON health response.

## Important next steps
1. Ask students and faculty to review career profiles and assessment questions.
2. Add privacy notice, deletion/export controls and proper authentication before collecting real student data.
3. Add automated tests.
4. Add ML only after you have an appropriate, permissioned dataset and a valid evaluation method. Do not claim invented data proves accuracy.
5. Add an AI assistant via a server-side API integration only after choosing a provider. Store API keys in environment variables, never in frontend JavaScript or Git.

## Current limitations
- This is a local educational prototype, not a validated psychometric assessment.
- It has no accounts or access control; anyone using the local instance can see the assessment history.
- `debug=True` is for local development only. Do not expose this development server publicly.
