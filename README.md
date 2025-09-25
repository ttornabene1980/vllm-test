
uv python pin 3.12
uv venv
uv pip install  -r pyproject.toml
uv sync
 

PYTHONPATH=. streamlit run  dashboard/app.py



Test
PYTHONPATH=. uv run pytest/allmini.py



rsync -azh /Users/tindarotornabene/develop/sorgente/vllm-test itoperator@10.199.145.180:/home/itoperator/test/