# How to Run Medical Chatbot Project

## Quick Start (3 Steps)

### Step 1: Open Terminal and Go to Project Folder
```bash
cd "c:\Users\VICTUS\Desktop\chatbot\Medical-Chatbot"
```

### Step 2: Activate Virtual Environment
**For PowerShell:**
```powershell
.\medibot\Scripts\Activate.ps1
```

**For Command Prompt:**
```cmd
medibot\Scripts\activate.bat
```

### Step 3: Run the Application
```bash
python app.py
```

### Step 4: Open Browser
Go to: **http://localhost:8080**

---

## Detailed Setup (First Time Only)

### Prerequisites
- Python 3.8 or higher
- Virtual environment (already created: `medibot/`)
- API keys in `.env` file (already configured)

### Step 1: Navigate to Project
```bash
cd "c:\Users\VICTUS\Desktop\chatbot\Medical-Chatbot"
```

### Step 2: Activate Virtual Environment
**For PowerShell:**
```powershell
.\medibot\Scripts\Activate.ps1
```

**For Command Prompt:**
```cmd
medibot\Scripts\activate.bat
```

You should see `(medibot)` at the start of your command prompt.

### Step 3: Install Dependencies
```bash
pip install -r config\requirements.txt
```

**Wait for installation to complete.**

### Step 4: Create Vector Database (First Time Only)
```bash
python scripts\store_index.py
```

This will:
- Load PDF documents from `data/` folder
- Process and split them into chunks
- Create embeddings
- Store in Pinecone vector database

**This step takes 5-10 minutes. Wait for it to finish.**

### Step 5: Run the Flask Application
```bash
python app.py
```

You should see:
```
* Serving Flask app 'app'
* Debug mode: on
* Running on http://127.0.0.1:8080
```

### Step 6: Open in Browser
Open your web browser and go to:
- **http://localhost:8080**
- or **http://127.0.0.1:8080**

### Step 7: Start Chatting
Type your medical questions in the chat interface!

---

## To Stop the Server
Press **Ctrl + C** in the terminal

---

## Common Commands Reference

| Task | Command |
|------|---------|
| Go to project | `cd "c:\Users\VICTUS\Desktop\chatbot\Medical-Chatbot"` |
| Activate (PowerShell) | `.\medibot\Scripts\Activate.ps1` |
| Activate (CMD) | `medibot\Scripts\activate.bat` |
| Install packages | `pip install -r config\requirements.txt` |
| Create database | `python scripts\store_index.py` |
| Run app | `python app.py` |
| Stop server | Press `Ctrl + C` |

---

## Troubleshooting

### Issue: "Module not found"
**Solution:** Install dependencies
```bash
pip install -r config\requirements.txt
```

### Issue: "API key error"
**Solution:** Check `.env` file has valid API keys

### Issue: "Port 8080 already in use"
**Solution:** Stop other apps using port 8080 or change port in `app.py`

### Issue: "Pinecone index not found"
**Solution:** Run the store_index script
```bash
python scripts\store_index.py
```

---

## Project Structure Quick Reference

```
Medical-Chatbot/
├── app.py                      ← Run this file
├── config\requirements.txt     ← Install dependencies
├── scripts\store_index.py      ← Create database
├── data\                       ← PDF files
├── src\                        ← Source code
├── docs\                       ← Documentation
└── medibot\                    ← Virtual environment
```

---

## Daily Use (After First Setup)

You only need to do this every time you want to run the project:

```bash
# 1. Go to folder
cd "c:\Users\VICTUS\Desktop\chatbot\Medical-Chatbot"

# 2. Activate environment
.\medibot\Scripts\Activate.ps1

# 3. Run app
python app.py

# 4. Open browser: http://localhost:8080
```

That's it! 🎉
