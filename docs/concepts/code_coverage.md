# Code Coverage

## Quick Commands

### Terminal report (per-file summary)

```bash
pytest tests/ --cov=src/llm_engine --cov-report=term-missing
```

- Shows % coverage per file
- `Missing` column shows exact uncovered line numbers

### Specific modules only

```bash
pytest tests/test_router.py tests/test_api_server.py --cov=src/llm_engine/serving --cov-report=term-missing
```

### Generate XML for VS Code Coverage Gutters

```bash
pytest tests/ --cov=src/llm_engine --cov-report=xml
```

- Creates `coverage.xml` in project root
- Open any source file → `Ctrl+Shift+P` → "Coverage Gutters: Display Coverage"
- Green = covered, Red = uncovered

### Generate HTML report

```bash
pytest tests/ --cov=src/llm_engine --cov-report=html
```

- Creates `htmlcov/` folder
- Open `htmlcov/index.html` in browser for line-by-line visual report

### Combined (terminal + XML + HTML)

```bash
pytest tests/ --cov=src/llm_engine --cov-report=term-missing --cov-report=xml --cov-report=html
```

## VS Code Setup

Add to `.vscode/settings.json`:

```json
"coverage-gutters.coverageBaseDir": ".",
"coverage-gutters.coverageFileNames": ["coverage.xml"]
```

## Dependencies

```bash
pip install pytest-cov
```

## .gitignore

```
htmlcov/
.coverage
coverage.xml
```
