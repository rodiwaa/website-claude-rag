install:
	uv venv .venv
	uv pip install -r requirements.txt --python .venv/bin/python3

run:
	.venv/bin/python3 main.py

run-ui:
	.venv/bin/chainlit run chainlit_app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; find . -name "*.pyc" -delete 2>/dev/null; true
