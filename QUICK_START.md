# Quick Start Guide - Medical Chatbot

## What Changed?

Your project is now **clean and organized**! All files are in proper folders.

## File Locations

| What You Need | Where to Find It |
|---------------|------------------|
| Main app | `app.py` (root folder) |
| Install packages | `config/requirements.txt` |
| Create database | `scripts/store_index.py` |
| Source code | `src/` folder |
| Documentation | `docs/` folder |
| Images/Graphs | `docs/` folder |
| Analysis scripts | `scripts/` folder |

## How to Run (3 Simple Steps)

### 1. Install Dependencies
```bash
pip install -r config/requirements.txt
```

### 2. Create Vector Database
```bash
python scripts/store_index.py
```

### 3. Start the Chatbot
```bash
python app.py
```

Then open: http://localhost:8080

## What's in Each Folder?

- **src/** - Core code (helper functions, prompts)
- **scripts/** - All Python scripts (analysis, setup)
- **config/** - Settings and requirements
- **docs/** - Documentation and images
- **data/** - PDF files (medical books)
- **research/** - Evaluation data and notebooks
- **templates/** - HTML pages
- **static/** - CSS styles

## Need Help?

Check `ORGANIZATION_GUIDE.md` for full details!
