# 🤖 Joi - Your English Practice AI Agent

<p align="center">
    <img alt="Joi Logo" src="img/joi_logo.png" width="200" />
    <h1 align="center">📱 Joi 📱</h1>
    <h3 align="center">Practice English Every Day on WhatsApp</h3>
</p>

<p align="center">
    <img alt="WhatsApp Logo" src="img/whatsapp_logo.png" width="100" />
</p>

## 🎯 What is Joi?

**Joi** is an AI-powered English practice agent that lives on WhatsApp. Designed to help you improve your English conversation skills through daily interactions, Joi makes language learning natural, convenient, and fun.

### ✨ What Joi Can Do:

- 💬 **Natural Conversations**: Chat in English about any topic
- 🎤 **Voice Practice**: Send voice messages and get audio responses
- 🖼️ **Visual Learning**: Share images and describe them in English
- 🧠 **Personalized Memory**: Remembers your progress and preferences
- 📚 **Smart Corrections**: Gentle feedback to improve your English
- ⚡ **Always Available**: Practice anytime, anywhere on WhatsApp

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- WhatsApp Business API access
- Supabase account (PostgreSQL database)
- Qdrant Cloud account (vector database)
- API keys for AI services (Groq, ElevenLabs, Together AI)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alcarrillop/joi.git
   cd joi
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database URLs
   ```

4. **Run the service**
   ```bash
   uvicorn src.agent.interfaces.whatsapp.webhook_endpoint:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Configure WhatsApp webhook**
   - Use ngrok for local testing: `ngrok http 8000`
   - Set webhook URL: `https://your-ngrok-url.ngrok.io/whatsapp_response`

---

## 🏗️ Architecture

### Core Components

```
src/agent/
├── core/
│   ├── database.py          # PostgreSQL checkpointer setup
│   ├── prompts.py          # AI conversation prompts
│   └── exceptions.py       # Error handling
├── modules/
│   ├── memory/             # Long-term & short-term memory
│   ├── speech/             # Voice processing (STT/TTS)
│   └── image/              # Image analysis & generation
├── graph/                  # LangGraph conversation workflows
└── interfaces/
    └── whatsapp/           # WhatsApp API integration
```

### Tech Stack

| Technology | Purpose |
|------------|---------|
| **LangGraph** | Conversation workflow orchestration |
| **Groq** | Fast LLM inference (Llama models) |
| **Supabase** | PostgreSQL database for conversations |
| **Qdrant Cloud** | Vector database for long-term memory |
| **ElevenLabs** | Text-to-Speech for voice responses |
| **Together AI** | Image generation capabilities |
| **FastAPI** | WhatsApp webhook server |

---

## 🔧 Configuration

### Required Environment Variables

```bash
# AI Services
GROQ_API_KEY=your_groq_api_key
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_voice_id
TOGETHER_API_KEY=your_together_api_key

# Databases
DATABASE_URL=postgresql://user:pass@host:port/db
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

# WhatsApp
WHATSAPP_TOKEN=your_whatsapp_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_VERIFY_TOKEN=your_verify_token
```

---

## 🎮 How to Use Joi

### For Learners

1. **Add Joi to WhatsApp** using the configured business number
2. **Start a conversation** - just say "Hello!"
3. **Practice daily**:
   - Send text messages for grammar practice
   - Send voice notes for pronunciation
   - Share images for vocabulary building
4. **Track progress** - Joi remembers your conversations and adapts

### For Developers

1. **Extend functionality** by adding new nodes to the LangGraph workflow
2. **Customize prompts** in `src/agent/core/prompts.py`
3. **Add new languages** by modifying the conversation logic
4. **Deploy to production** using Docker and Cloud Run

---

## 🚀 Deployment

### Local Development
```bash
uvicorn src.agent.interfaces.whatsapp.webhook_endpoint:app --reload
```

### Production Deployment
```bash
# Using Docker
docker build -t joi .
docker run -p 8000:8000 joi

# Or deploy to Google Cloud Run
gcloud run deploy joi --source .
```

---

## 🤝 Contributing

We welcome contributions! Whether you want to:
- Add new conversation features
- Improve the AI responses
- Add support for more languages
- Enhance the learning experience

Please feel free to open issues and pull requests.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for conversation workflows
- Powered by [Groq](https://groq.com/) for fast AI inference
- Voice synthesis by [ElevenLabs](https://elevenlabs.io/)
- Vector search by [Qdrant](https://qdrant.tech/)

---

<p align="center">
    <strong>Ready to practice English with Joi? 🚀</strong><br>
    <em>Your AI-powered English conversation partner is just a WhatsApp message away!</em>
</p>