# 🤖 Multi-Tool Autonomous AI Agent

### *A production-grade LangChain ReAct agent with real-time web search, safe math computation, local file reading & modern interactive Web UI*

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.4%2B-green?logo=chainlink)](https://langchain.com)
[![Flask](https://img.shields.io/badge/Flask-3.1%2B-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-42%20Passing-brightgreen?logo=pytest)](tests/)
[![Internship](https://img.shields.io/badge/Internship-Python%20%2B%20LangChain-purple)](.)

---

> **Task 3 · Python + LangChain Internship Program**  
> *"Implement a LangChain or function-calling agent equipped with tools: real-time web search (DuckDuckGo/Tavily), a Python math calculator, and local file reader. Display the reasoning chain (thought → action → observation → final answer)."*

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Tools Overview](#-tools-overview)
- [Quick Start](#-quick-start)
- [Web UI & REST API](#-web-ui--rest-api)
- [Configuration](#-configuration)
- [Example Queries](#-example-queries)
- [Running Tests](#-running-tests)
- [Performance & Latency Optimizations](#-performance--latency-optimizations)
- [License](#-license)

---

## 🎯 Overview

This project implements an **autonomous ReAct (Reasoning + Acting) AI Agent** using LangChain. Rather than using fixed workflows, the agent actively inspects user requests, devises a multi-step solution, selects the appropriate tools, and executes them step-by-step:

```
Thought -> Action -> Action Input -> Observation -> ... -> Final Answer
```

### Both Interfaces Supported
1. **Interactive Web UI**: Modern dark-mode interface with live SSE streaming and step cards.
2. **Terminal Console REPL**: Color-coded CLI with full reasoning chain display.

---

## ✨ Key Features

- **⚡ Fast Inference**: Pre-configured with Google Gemini Flash for sub-3-second responses.
- **🔍 Real-Time Web Search**: DuckDuckGo integration (free, no API key needed) + optional Tavily AI search.
- **🧮 Sandboxed Math Calculator**: Safe Python `eval()` whitelist environment with zero OS access, plus SymPy for symbolic algebra.
- **📄 Local File Reader**: Supports `.txt`, `.md`, `.csv`, `.json`, `.pdf`, and `.docx` with path traversal protections.
- **🌐 Full Web UI**: Flask-powered interface with Server-Sent Events (SSE) for streaming thoughts and actions in real time.
- **🧪 Production Test Suite**: 42 unit tests with pytest covering arithmetic, security boundaries, and document loaders.

---

## 🏗 Architecture

```
                      ┌────────────────────────┐
                      │      USER REQUEST      │
                      │  (Web UI or Terminal)  │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │     AgentExecutor      │
                      │     (ReAct Loop)       │
                      └───────────┬────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 🔍 web_search    │    │ 🧮 calculator    │    │ 📄 file_reader   │
│ DuckDuckGo/Tavily│    │ Math + SymPy     │    │ TXT/JSON/PDF/DOC │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

---

## 📁 Project Structure

```
multi-tool-autonomous-ai-agent/
│
├── main.py                     # CLI Interactive Console REPL
├── demo.py                     # Automated 5-step test script
├── app.py                      # Flask Web Application & SSE Server
├── requirements.txt            # Project dependencies
├── .env.example                # Configuration template
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
│
├── agent/                      # Core Agent Package
│   ├── __init__.py
│   ├── agent_builder.py        # ReAct AgentExecutor & memory setup
│   └── llm_factory.py          # Multi-provider LLM Factory (Gemini, OpenAI, Claude)
│
├── tools/                      # Agent Tools
│   ├── __init__.py             # Tool registry (get_all_tools)
│   ├── search_tool.py          # DuckDuckGo & Tavily web search
│   ├── calculator_tool.py      # Safe math evaluator & SymPy
│   └── file_reader_tool.py     # Local file parser (TXT, JSON, PDF, DOCX)
│
├── utils/                      # Utilities
│   ├── __init__.py
│   └── display.py              # CLI color formatting & output rendering
│
├── templates/                  # Frontend Templates
│   └── index.html              # Chat UI layout
│
├── static/                     # Frontend Assets
│   ├── css/style.css           # Modern dark-mode styling
│   └── js/app.js               # EventSource client & dynamic UI renderer
│
├── tests/                      # Unit Tests
│   ├── __init__.py
│   ├── test_calculator.py      # 31 calculator & security unit tests
│   └── test_file_reader.py     # 11 file reader & path safety unit tests
│
└── sample_data/                # Sample Files for Testing
    ├── employees.csv
    └── research_notes.md
```

---

##Screenshot

<img width="1917" height="902" alt="multi" src="https://github.com/user-attachments/assets/d2e7e345-5358-485e-b478-ba0a094fe122" />

---


---

## 🛠 Tools Overview

### 1. Web Search Tool (`web_search`)
- Searches the live internet using **DuckDuckGo** (default, no API key required) or **Tavily**.
- Limits output to top 3 relevant results for faster agent reasoning.

### 2. Python Calculator (`python_calculator`)
- Evaluates arithmetic expressions, trigonometry, powers, roots, and logarithms.
- Supports symbolic algebra via SymPy (`solve`, `diff`, `integrate`).
- **Security Guardrail**: Runs in a sandboxed namespace with `__builtins__ = {}`. Blocks `import`, `exec`, `eval`, `open`, and system libraries.

### 3. File Reader (`file_reader`)
- Reads and parses local documents: `.txt`, `.md`, `.csv`, `.json`, `.pdf`, `.docx`.
- **Security Guardrail**: Detects and rejects path traversal sequences (`..`). Implements file size limits (5 MB) and character truncation.

---

## 🚀 Quick Start

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/your-username/multi-tool-autonomous-ai-agent.git
cd multi-tool-autonomous-ai-agent

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Copy the environment template and insert your free Gemini API key:

```bash
copy .env.example .env
```

Edit `.env`:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
LLM_PROVIDER=google
LLM_MODEL=gemini-3.5-flash-lite
SEARCH_PROVIDER=duckduckgo
```

> **Get a free Google Gemini key**: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

## 🌐 Web UI & REST API

### Launch the Web UI

```bash
python app.py
```

Open your browser to: **`http://localhost:5000`**

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the web interface |
| `/api/status` | GET | Reports health and registered tools |
| `/api/chat` | POST | Submits a query; returns JSON with answer & steps |
| `/api/stream` | GET | Server-Sent Events (SSE) live-streaming endpoint |
| `/api/clear` | POST | Clears the agent's conversation memory |

---

## 💻 Terminal CLI Mode

To interact with the agent via command line:

```bash
python main.py
```

### Console Commands
- `/help`: Show command guide
- `/tools`: List registered tools
- `/clear`: Reset agent conversation history
- `/exit`: Quit session

---

## 💡 Example Queries

| Intent | Sample Prompt |
|--------|---------------|
| **Web Search** | `"What are the latest AI hardware announcements in 2025?"` |
| **Math Computation** | `"Calculate compound interest on $10,000 at 7% for 5 years compounded monthly"` |
| **Symbolic Algebra** | `"Solve the quadratic equation x^2 - 7x + 12 = 0"` |
| **File Reading** | `"Summarize the employees.csv file located in sample_data"` |
| **Compound / Multi-Tool** | `"Find the speed of light in km/s and calculate how many seconds it takes to reach the Moon (384,400 km)"` |

---

## ✅ Running Tests

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --cov=.
```

**Result:** 42 passed tests verifying arithmetic precision, symbolic solving, forbidden keyword rejection, file handling, and security boundaries.

---

## ⚡ Performance & Latency Optimizations

1. **Optimized Model Selection**: Updated to `gemini-3.5-flash-lite`, reducing latency from 40+ seconds (caused by deprecated 2.0 endpoint retries) to **~2.5 seconds**.
2. **Compact ReAct Prompting**: Concise system instructions prevent verbosity and reduce token generation overhead.
3. **DuckDuckGo Rate Management**: Results bounded to top 3 relevant entries to minimize prompt bloat.
4. **Live SSE Streaming**: Web UI uses Server-Sent Events to render agent tool calls as they occur.

---

## 📄 License

This project is licensed under the **MIT License**.
