PYTHON = python3
PIP = pip3


run: 
	uv run $(PYTHON) -m src

clean:
	rm -rf src/__pycache__/