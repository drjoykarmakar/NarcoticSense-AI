# Upload NarcoticSense AI to GitHub

## Option A: GitHub website

1. Create a new GitHub repository named `narcoticsense-ai`.
2. Keep it public if you want an open-source project.
3. Do not add a README/license from GitHub because this folder already has them.
4. Upload all files from this folder.
5. Commit with the message: `Initial NarcoticSense AI platform`.

## Option B: Terminal

```bash
git init
git add .
git commit -m "Initial NarcoticSense AI platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/narcoticsense-ai.git
git push -u origin main
```

## Run locally on Mac

Double-click:

```text
START_HERE.command
```

Manual method:

```bash
python3 -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python3 -m streamlit run app/streamlit_app.py
```

## Developer checks

```bash
python3 -m pip install -e ".[dev]"
python3 -m ruff check src tests app
python3 -m black --check -W 1 src tests app
python3 -m pytest -q
```
