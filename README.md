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
   
   The repository includes a `.env.example` file with placeholders for all
   configuration options defined in `src/agent/settings.py`. Use it as the
   starting point for your own environment file.
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database URLs
   # Never commit your filled .env file to version control
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

- **Database**: PostgreSQL with async access via asyncpg
- **Users**: Example data with a handful of test users
- **Messages**: Persisted conversations in the database
- **Memory**: Long term storage using Qdrant vector store
- **Curriculum**: CEFR-aligned competencies (A1-C2 levels)
- **Assessment**: Real-time evaluation with grammar and vocabulary metrics
- **Performance**: Target response time under 10 seconds
- **Test Coverage**: Work in progress

### System Features

- 🇨🇴 **Real User**: Colombian user practicing English with girlfriend context
- 🎓 **Adaptive Learning**: A1 level user with 3 completed competencies  
- 📊 **Smart Assessment**: Grammar, vocabulary, fluency analysis
- 🧠 **Context Memory**: Remembers personal details, preferences, relationships
- 🔧 **Debug Interface**: Real-time monitoring and analytics

---

## 🔧 Configuration

The repository includes a `.env.example` file with a complete list of
configuration variables. Copy it and update the values to match your setup.

### Required Environment Variables

```bash
DATABASE_URL=postgresql://...          # Supabase
QDRANT_URL=https://...                # Qdrant Cloud  
QDRANT_API_KEY=...                    # Qdrant API key
WHATSAPP_TOKEN=...                    # WhatsApp Business API
WHATSAPP_PHONE_NUMBER_ID=...          # WhatsApp Phone Number
WHATSAPP_WEBHOOK_VERIFY_TOKEN=...     # Webhook verification
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
pytest -q
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

## 🚀 Deployment

### System Status
- **Memory System**: Uses Qdrant vector store
- **Database**: 8 essential tables
- **Learning Analytics**: Vocabulary & grammar tracking active
- **Performance**: Averages under 10 seconds per message
- **Architecture**: LangGraph workflow with persistent memory

## 🎯 Quick Deploy to Railway

1. **Connect your repo** to Railway
2. **Set environment variables** (see `DEPLOYMENT.md`)
3. **Deploy** - Railway handles the rest!

### Required Environment Variables
```bash
DATABASE_URL=postgresql://...          # Supabase
QDRANT_URL=https://...                # Qdrant Cloud  
QDRANT_API_KEY=...                    # Qdrant API key
WHATSAPP_TOKEN=...                    # WhatsApp Business API
WHATSAPP_PHONE_NUMBER_ID=...          # WhatsApp Phone Number
WHATSAPP_WEBHOOK_VERIFY_TOKEN=...     # Webhook verification
```

## 🏗️ Architecture

```
WhatsApp → FastAPI → LangGraph → Memory System → AI Response
                        ↓
               Supabase (PostgreSQL) + Qdrant (Vectors)
```

### Core Components
- **Memory Manager**: Semantic search with user isolation
- **Learning Stats**: Vocabulary and grammar progress tracking  
- **Curriculum Manager**: Dynamic level estimation
- **Multi-modal Support**: Text, audio, and image processing

## 📊 Database Schema (Simplified)

| Table | Purpose |
|-------|---------|
| `users` | Basic user management (id, phone, name) |
| `sessions` | Message grouping containers |
| `messages` | Complete conversation history |
| `learning_stats` | Vocabulary & grammar progress |
| `checkpoint_*` | LangGraph workflow state |

## 🔍 Debug & Monitoring

| Endpoint | Purpose |
|----------|---------|
| `/debug/health` | System health check |
| `/debug/stats` | System statistics |
| `/debug/dashboard` | Web dashboard |
| `/debug/users` | User management |

## 🛠️ Local Development

```bash
# Clone and setup
git clone <repo>
cd joi

# Install dependencies
uv sync

# Set environment variables
cp .env.example .env  # Edit with your values

# Run locally
fastapi dev src/agent/interfaces/whatsapp/webhook_endpoint.py

# Test with Docker
docker compose up --build
```

## 📈 Performance Metrics

- **Message Processing**: ~9 seconds average
- **Memory Retrieval**: <1 second semantic search
- **Learning Analytics**: <2 seconds vocabulary analysis
- **System Health**: All components operational

## 🔧 Tech Stack

- **Framework**: FastAPI + LangGraph  
- **AI**: Groq LLM models
- **Memory**: Qdrant vector database
- **Database**: PostgreSQL (Supabase)
- **Deployment**: Railway + Docker
- **Language**: Python 3.12 with uv

## 📚 Key Features

### Memory System
- ✅ Semantic search of conversation history
- ✅ User-specific memory isolation  
- ✅ Automatic importance filtering
- ✅ GDPR-compliant cleanup

### Learning Analytics  
- ✅ Vocabulary progress tracking
- ✅ Grammar error analysis
- ✅ Dynamic level estimation
- ✅ Personalized feedback

### Multi-modal Support
- ✅ Text conversations
- ✅ Audio transcription
- ✅ Image analysis
- ✅ WhatsApp integration

## 🔐 Security & Production

- ✅ Environment-based configuration
- ✅ Health checks and monitoring
- ✅ Non-root Docker user
- ✅ Input validation and error handling
- ✅ Structured logging
- ✅ Secrets managed via environment variables

## 📖 Documentation

- `DEPLOYMENT.md` - Complete deployment guide
- `/debug/dashboard` - Interactive system overview
- `/debug/health` - Real-time health status

---

**Ready for real users!** 🎉 Deploy to Railway and start teaching languages with AI-powered memory.