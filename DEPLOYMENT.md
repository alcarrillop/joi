# JOI - Railway Deployment Guide

## 🚀 Quick Deploy to Railway

### Prerequisites
- Supabase PostgreSQL database (already configured)
- Qdrant Cloud vector database (already configured)
- WhatsApp Business API access
- OpenAI API key

### 1. Environment Variables

Set these in Railway dashboard:

```bash
# Database (Supabase)
DATABASE_URL=postgresql://user:password@host:port/database

# Vector Database (Qdrant Cloud)  
QDRANT_URL=https://your-cluster-url
QDRANT_API_KEY=your-api-key

# WhatsApp API
WHATSAPP_TOKEN=your-access-token
WHATSAPP_PHONE_NUMBER_ID=your-phone-number-id
WHATSAPP_WEBHOOK_VERIFY_TOKEN=your-webhook-verify-token

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Environment
TESTING=false
```

### 2. Deploy Steps

1. **Connect Repository**: Link your GitHub repo to Railway
2. **Set Environment Variables**: Add all variables above in Railway dashboard
3. **Deploy**: Railway will automatically build using `Dockerfile`
4. **Health Check**: Service will be available at `/debug/health`

### 3. WhatsApp Webhook Configuration

After deployment, configure your WhatsApp webhook URL:
```
https://your-railway-app.railway.app/whatsapp_response
```

### 4. System Architecture

```
WhatsApp → Railway App → Supabase (PostgreSQL) + Qdrant (Vector DB)
```

### 5. Local Testing

Test locally before deployment:

```bash
# Create .env file with your variables
cp .env.example .env

# Build and run with Docker Compose
docker-compose up --build

# Test health endpoint
curl http://localhost:8000/debug/health
```

### 6. Monitoring

- **Health Check**: `/debug/health`
- **System Stats**: `/debug/stats` 
- **Debug Dashboard**: `/debug/dashboard`
- **Users**: `/debug/users`

### 7. Database Status

✅ **Simplified & Production Ready**:
- 8 essential tables (down from 22)
- Memory-focused architecture
- LangGraph-aligned sessions
- No unnecessary fields

### 8. Performance

- **Processing**: ~9 seconds per message
- **Memory Retrieval**: <1 second
- **Learning Analytics**: <2 seconds

---

**Ready for Production!** 🎯 