NarcoticSense AI - Mac instructions

1. Unzip this folder.
2. Open the folder.
3. Double-click START_HERE.command.
4. If macOS blocks it, right-click START_HERE.command and choose Open.

Do not run pip install -e . for this beginner version.
Do not use an old copied folder. Use only this folder.

Manual terminal method:

cd /path/to/narcoticsense-ai
python3 -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python3 -m streamlit run "$PWD/app/streamlit_app.py"
