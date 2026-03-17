install:
	uv venv .venv
	uv pip install -r requirements.txt --python .venv/bin/python3

run:
	.venv/bin/python3 main.py

run-ui:
	.venv/bin/chainlit run chainlit_app.py

eval:
	uv run python run_evals.py

eval-fresh:
	uv run python run_evals.py --fresh

install-hooks:
	cp scripts/pre-commit .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit

skip-commit:
	SKIP_EVALS=1 git commit $(ARGS)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; find . -name "*.pyc" -delete 2>/dev/null; true
