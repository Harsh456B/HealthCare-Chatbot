# Project Organization Guide

## New Folder Structure

Your Medical-Chatbot project is now organized into clear, logical folders:

### 📁 Root Directory
- **app.py** - Main Flask application (keep in root)
- **setup.py** - Package configuration (keep in root)
- **.env** - Environment variables (keep in root)
- **LICENSE** - Project license (keep in root)
- **README.md** - Main documentation (keep in root)

### 📁 src/ (Source Code)
Contains the core application code:
- `__init__.py` - Package initializer
- `helper.py` - Document processing functions
- `prompt.py` - AI prompt templates

### 📁 scripts/ (Utility Scripts)
Contains all Python scripts for analysis and utilities:
- `store_index.py` - Creates Pinecone vector index
- `template.sh` - Shell template
- All analysis scripts (moved from root)

### 📁 config/ (Configuration)
Contains configuration files:
- `requirements.txt` - Python dependencies

### 📁 docs/ (Documentation)
Contains all documentation and images:
- All `.md` files (guides, documentation)
- All `.png` files (graphs, charts, diagrams)

### 📁 research/ (Research Data)
Contains research and evaluation data:
- `retrieval_evaluation_results.csv` - Evaluation results

### 📁 data/ (Data Files)
Contains PDF documents for the chatbot:
- `Medical_book.pdf` - Medical reference book

### 📁 templates/ (HTML Templates)
- `chat.html` - Chat interface

### 📁 static/ (Static Files)
- `style.css` - CSS styles

### 📁 tests/ (Future Use)
Ready for test files

### 📁 medibot/ (Virtual Environment)
Python virtual environment (do not modify)

## How to Use

### Running the Application
```bash
# Install dependencies
pip install -r config/requirements.txt

# Create vector index
python scripts/store_index.py

# Run the app
python app.py
```

### Adding New Files
- **Scripts** → Put in `scripts/` folder
- **Documentation** → Put in `docs/` folder
- **Configuration** → Put in `config/` folder
- **Images/Graphs** → Put in `docs/` folder
- **Data files** → Put in `data/` folder

## Benefits of This Structure

✅ **Easy Navigation** - Find files quickly by type
✅ **Clean Root** - Only essential files in root directory
✅ **Logical Grouping** - Related files are together
✅ **Scalable** - Easy to add new features
✅ **Professional** - Follows industry standards

## Quick Reference

| File Type | Location |
|-----------|----------|
| Main app | Root (app.py) |
| Source code | src/ |
| Scripts | scripts/ |
| Dependencies | config/requirements.txt |
| Documentation | docs/ |
| Images | docs/ |
| Data | data/ |
| HTML templates | templates/ |
| CSS/JS | static/ |
