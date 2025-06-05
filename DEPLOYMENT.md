# JOI English Agent - Railway Deployment Guide

## 🚀 Quick Deploy to Railway

### Prerequisites
- Supabase PostgreSQL database (already configured)
- Qdrant Cloud vector database (already configured)
- WhatsApp Business API access
- Railway account and CLI

### 1. Environment Variables

Set these in Railway dashboard:

```bash
# Core Configuration
ENVIRONMENT=production
PORT=8000

# API Keys
GROQ_API_KEY=your_groq_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
TOGETHER_API_KEY=your_together_api_key_here

# Database (Supabase)
DATABASE_URL=postgresql://user:password@host:port/database
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here

# Vector Database (Qdrant Cloud)  
QDRANT_URL=https://your-cluster-url
QDRANT_API_KEY=your-api-key

# WhatsApp API
WHATSAPP_ACCESS_TOKEN=your-access-token
WHATSAPP_PHONE_NUMBER_ID=your-phone-number-id
WHATSAPP_WEBHOOK_VERIFY_TOKEN=your-webhook-verify-token
```

### 2. CI/CD Pipeline

The project includes automated CI/CD with:
- **Linting**: Ruff formatting and type checking
- **Testing**: pytest with coverage reports  
- **Security**: Trivy vulnerability scanning
- **Docker Build**: Multi-stage production image
- **Auto Deploy**: Pushes to main branch deploy to Railway

**GitHub Secrets Required**:
- `RAILWAY_TOKEN`: Your Railway API token

### 3. Deploy Steps

#### Option A: Automatic (Recommended)
1. **Push to main**: CI/CD will automatically deploy
2. **Monitor**: Check GitHub Actions for deployment status

#### Option B: Manual
1. **Connect Repository**: Link your GitHub repo to Railway
2. **Set Environment Variables**: Add all variables above in Railway dashboard  
3. **Deploy**: Railway will automatically build using `Dockerfile`
4. **Health Check**: Service will be available at `/health`

### 4. Docker Production Fixes

✅ **Fixed Issues**:
- **Port Configuration**: Now uses Railway's `$PORT` environment variable
- **Import Path**: Corrected to `src.agent.interfaces.whatsapp.webhook_endpoint:app`
- **Health Check**: Extended timeout (60s) for Railway startup
- **Security**: Runs as non-root user with proper permissions
- **SSL/TLS**: Updated CA certificates for secure connections

### 5. WhatsApp Webhook Configuration

After deployment, configure your WhatsApp webhook URL:
```
https://your-railway-app.railway.app/whatsapp_response
```

### 6. System Architecture

```
WhatsApp → Railway App → Supabase (PostgreSQL) + Qdrant (Vector DB)
```

### 7. Local Development

Test locally before deployment:

```bash
# Install dependencies
uv sync --dev

# Create .env file with your variables  
cp .env.example .env

# Run locally
uv run uvicorn src.agent.interfaces.whatsapp.webhook_endpoint:app --reload

# Or test with Docker
docker build -t joi-english-agent .
docker run -p 8000:8000 --env-file .env joi-english-agent

# Test health endpoint
curl http://localhost:8000/health
```

### 8. Monitoring & Debugging

- **Health Check**: `/health`
- **System Stats**: `/debug/stats` (if available)
- **Debug Dashboard**: `/debug/dashboard` (if available)
- **Railway Logs**: Monitor in Railway dashboard

### 9. Database Status

✅ **Simplified & Production Ready**:
- 8 essential tables (down from 22)
- Memory-focused architecture
- LangGraph-aligned sessions
- No unnecessary fields

### 10. Performance Targets

- **Processing**: ~9 seconds per message
- **Memory Retrieval**: <1 second
- **Learning Analytics**: <2 seconds
- **Health Check**: <5 seconds

### 11. Troubleshooting

**Common Issues**:
1. **Port binding**: Ensure `PORT` env var is set in Railway
2. **Import errors**: Verify `src.agent.interfaces.whatsapp.webhook_endpoint:app` path
3. **Health check timeout**: Extended to 60s for Railway startup
4. **SSL issues**: Updated CA certificates in Docker image

**Logs & Debugging**:
```bash
# Check Railway logs
railway logs

# Local debug
docker logs <container_id>
```

---

**✅ Production Ready!** 🎯 

The deployment is now optimized for Railway with proper CI/CD, security scanning, and automated testing. 