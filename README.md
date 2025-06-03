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

**Joi** is an AI-powered English learning assistant that lives on WhatsApp. Designed to help you improve your English conversation skills through daily interactions, Joi makes language learning natural, convenient, and fun with adaptive curriculum and real-time assessment.

### ✨ What Joi Can Do:

- 💬 **Natural Conversations**: Chat in English about any topic with contextual responses
- 🎤 **Voice Practice**: Send voice messages and get audio responses 
- 🖼️ **Visual Learning**: Share images and describe them in English
- 🧠 **Smart Memory**: Remembers your progress, preferences, and personal details
- 📚 **Adaptive Curriculum**: CEFR-aligned learning path (A1-C2) with personalized recommendations
- 📊 **Real-time Assessment**: Automatic evaluation of grammar, vocabulary, and fluency
- 🔧 **Debug Tools**: Comprehensive monitoring and analytics
- ⚡ **Always Available**: Practice anytime, anywhere on WhatsApp

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- WhatsApp Business API access
- PostgreSQL database
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

4. **Initialize database**
   ```bash
   python scripts/create_curriculum_tables.py
   python scripts/create_assessment_tables.py
   python scripts/create_missing_tables.py
   ```

5. **Run the service**
   ```bash
   uvicorn src.agent.interfaces.whatsapp.webhook_endpoint:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Configure WhatsApp webhook**
   - Use ngrok for local testing: `ngrok http 8000`
   - Set webhook URL: `https://your-ngrok-url.ngrok.io/whatsapp_response`

---

## 🏗️ Architecture

### Core Components

```
src/agent/
├── core/
│   ├── database.py          # PostgreSQL connection and setup
│   ├── prompts.py          # AI conversation prompts
│   └── exceptions.py       # Error handling
├── modules/
│   ├── memory/             # Long-term & short-term memory system
│   ├── speech/             # Voice processing (STT/TTS)
│   ├── image/              # Image analysis & generation
│   ├── curriculum/         # CEFR curriculum management
│   └── assessment/         # Real-time language assessment
├── graph/                  # LangGraph conversation workflows
└── interfaces/
    └── whatsapp/           # WhatsApp API integration

test/                       # Comprehensive test suite
scripts/                    # Database setup and debug tools
docs/                       # Detailed system documentation
```

### Tech Stack

| Technology | Purpose |
|------------|---------|
| **LangGraph** | Conversation workflow orchestration |
| **Groq** | Fast LLM inference (Llama models) |
| **PostgreSQL** | Main database for users, sessions, curriculum |
| **Qdrant Cloud** | Vector database for long-term memory |
| **ElevenLabs** | Text-to-Speech for voice responses |
| **Together AI** | Image generation capabilities |
| **FastAPI** | WhatsApp webhook server |

---

## 📚 System Status

### Current Production State ✅

- **Database**: 27 tables, 6 views, proper foreign key constraints
- **Users**: 3 active users with real conversation data
- **Messages**: 54+ messages processed and stored
- **Memory**: 21 personalized memories extracted
- **Curriculum**: CEFR-aligned competencies (A1-C2 levels)
- **Assessment**: Real-time evaluation with 14 skill metrics
- **Performance**: All endpoints responding <5 seconds
- **Test Coverage**: 100% system validation (14/14 components)

### System Features

- 🇨🇴 **Real User**: Colombian user practicing English with girlfriend context
- 🎓 **Adaptive Learning**: A1 level user with 3 completed competencies  
- 📊 **Smart Assessment**: Grammar, vocabulary, fluency analysis
- 🧠 **Context Memory**: Remembers personal details, preferences, relationships
- 🔧 **Debug Interface**: Real-time monitoring and analytics

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
4. **Track progress** - Joi adapts to your CEFR level and learning goals

### For Developers

1. **Run comprehensive tests**:
   ```bash
   python test/comprehensive_system_test.py  # 14-component validation
   python test/test_curriculum.py           # Curriculum system
   python test/test_assessment_system.py    # Assessment engine
   ```

2. **Monitor system health**:
   ```bash
   python scripts/debug_tools.py           # Interactive monitoring
   ```

3. **Extend functionality** by adding new nodes to the LangGraph workflow
4. **Customize curriculum** in the curriculum management system

---

## 📖 Documentation

Detailed documentation is available in the `/docs` folder:

- **[📊 Assessment System](docs/ASSESSMENT_SYSTEM.md)** - Real-time language evaluation
- **[🎓 Curriculum System](docs/CURRICULUM_SYSTEM.md)** - CEFR-aligned learning paths  
- **[🔧 Debug Tools](docs/DEBUG_TOOLS.md)** - Monitoring and analytics

### Testing & Scripts

- **[🧪 Test Suite](test/README.md)** - Comprehensive system validation
- **[⚙️ Scripts](scripts/README.md)** - Database setup and utilities

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

### Health Check
```bash
curl http://localhost:8000/debug/health
```

---

## 🧪 Testing

### Quick System Validation
```bash
# Test all 14 components
python test/comprehensive_system_test.py

# Expected output: 100% success rate (14/14 components)
```

### Component Tests
```bash
python test/test_curriculum.py      # Curriculum functionality
python test/test_assessment_system.py  # Assessment engine
python test/test_connections.py     # Database/API connections
```

---

## 🤝 Contributing

We welcome contributions! Whether you want to:
- Add new conversation features
- Improve the AI responses  
- Enhance the curriculum system
- Add new assessment metrics
- Improve the learning experience

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
- Curriculum based on [CEFR framework](https://www.coe.int/en/web/common-european-framework-reference-languages)

---

<p align="center">
    <strong>Ready to practice English with Joi? 🚀</strong><br>
    <em>Your AI-powered English learning assistant is just a WhatsApp message away!</em>
</p>