# Qora Assistant - LangGraph + Ollama Chatbot

A modern, agentic chatbot built with Streamlit, LangGraph, and Ollama Cloud.

## Quick Start (Local)

1. Create and activate the virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies:
   ```powershell
   .\.venv\Scripts\pip.exe install -r requirements.txt
   ```

3. Configure Ollama Cloud (see below) and launch:
   ```powershell
   .\.venv\Scripts\streamlit.exe run app.py
   ```

The app opens at `http://localhost:8501`. When running, Streamlit also shows a **Network URL** (e.g., `http://192.168.1.5:8501`) that others on your network can use.

## Ollama Cloud Configuration

### Option 1: Environment Variables (Recommended for Deployment)

Create a `.env` file in the project root:

```env
CHATBOT_OLLAMA_HOST=https://ollama.com
CHATBOT_OLLAMA_MODEL=glm-4.6:cloud
CHATBOT_OLLAMA_API_KEY=your-api-key-here
```

### Option 2: Windows System Variables

```powershell
setx CHATBOT_OLLAMA_HOST "https://ollama.com"
setx CHATBOT_OLLAMA_MODEL "glm-4.6:cloud"
setx CHATBOT_OLLAMA_API_KEY "your-api-key-here"
```

**Note:** After using `setx`, restart your terminal/PowerShell window.

## Sharing Your Chatbot

### Method 1: Share Locally (Temporary)

1. Run the app locally:
   ```powershell
   .\.venv\Scripts\streamlit.exe run app.py
   ```

2. Find the **Network URL** in the terminal output (e.g., `http://192.168.1.5:8501`).

3. Share that URL with others on the same network.

4. **Important:** Your computer must stay on and connected to the network for others to access it.

### Method 2: Deploy to Streamlit Cloud (Free & Permanent)

1. **Push your code to GitHub:**
   ```powershell
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/chatbot.git
   git push -u origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.

3. **Click "New app"** and fill in:
   - **Repository:** `yourusername/chatbot`
   - **Branch:** `main`
   - **Main file:** `app.py`
   - **App URL:** Choose a custom subdomain (optional)

4. **Add secrets** (Settings → Secrets) for Ollama Cloud:
   ```toml
   CHATBOT_OLLAMA_HOST = "https://ollama.com"
   CHATBOT_OLLAMA_MODEL = "glm-4.6:cloud"
   CHATBOT_OLLAMA_API_KEY = "your-api-key-here"
   ```

5. **Click "Deploy"** — your app will be live at `https://your-app-name.streamlit.app`

### Method 3: Deploy to Other Platforms

**Heroku:**
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
heroku config:set CHATBOT_OLLAMA_HOST=https://ollama.com
heroku config:set CHATBOT_OLLAMA_MODEL=glm-4.6:cloud
heroku config:set CHATBOT_OLLAMA_API_KEY=your-api-key
git push heroku main
```

**Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**AWS/Azure/GCP:** Use container services or App Services with similar configuration.

## Features

- 🤖 **Agentic behavior** using LangGraph for reasoning and routing
- 💬 **Clean, ChatGPT-like UI** with chat history
- 🧠 **Visible thinking process** (collapsible)
- 💾 **Persistent chat memory** across sessions
- 📋 **Copyable code blocks** with download support
- ⚙️ **Dynamic model/host switching** from the UI
- 🎨 **Modern, responsive design**

## Project Structure

```
Chatbot/
├── app.py                 # Streamlit frontend
├── backend/
│   ├── langgraph_agent.py # LangGraph agent implementation
│   ├── config.py          # Settings and environment variables
│   └── memory_manager.py  # Chat persistence
├── data/
│   └── chat_memory.json   # Saved chat history
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Troubleshooting

**"Could not reach Ollama host"**
- Verify your API key is correct
- Ensure `CHATBOT_OLLAMA_HOST` is set to `https://ollama.com` (not `/api`)
- Check your internet connection

**"Model not found"**
- Confirm the model name (e.g., `glm-4.6:cloud`) is available on your Ollama Cloud account
- Verify your API key has access to the model

**Chat history not persisting**
- Check that `data/chat_memory.json` exists and is writable
- Ensure the `data/` directory has proper permissions

## License

Built by Pratik — Feel free to use and modify for your projects!
