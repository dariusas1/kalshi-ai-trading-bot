# Railway Deployment Guide

## Prerequisites

1. **Railway Account**: Create account at [railway.app](https://railway.app)
2. **GitHub Repository**: Code must be pushed to GitHub
3. **Environment Variables**: Configure API keys in Railway

## Configuration Files Created

### `railway.json`
- Uses Nixpacks builder for Python 3.13
- Health check configuration
- Automatic restart on failure

### `Procfile`
- Defines web process command
- Runs the main trading bot

### `nixpacks.toml`
- Python 3.13 specification
- Dependency installation
- Virtual environment setup

### `.env.railway`
- Template for required environment variables
- Copy to Railway project settings

## Environment Variables Required

Add these in Railway project settings:

### API Keys
```
KALSHI_API_KEY=your_kalshi_api_key_here
XAI_API_KEY=your_xai_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### Trading Configuration
```
LIVE_TRADING_ENABLED=false  # Start with false for testing
```

### Database
```
DATABASE_URL=sqlite:///data/trading_system.db
```

### Optional: Kalshi Private Key
```
KALSHI_PRIVATE_KEY="[base64_encoded_private_key]"
```

**Important**: The code expects `KALSHI_PRIVATE_KEY` (uppercase) with base64-encoded PEM content. To encode your private key:

```bash
# Encode your private key to base64 (remove newlines)
cat kalshi_private_key.pem | base64 -w 0
```

Then set the encoded string as the environment variable.

## Deployment Steps

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add Railway deployment configuration"
   git push origin main
   ```

2. **Create Railway Project**:
   - Go to Railway dashboard
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

3. **Configure Environment**:
   - Go to project settings
   - Add all environment variables from `.env.railway`
   - Set `LIVE_TRADING_ENABLED=false` initially

4. **Deploy**:
   - Railway will automatically detect and deploy
   - Monitor logs for any issues

## Railway-Specific Considerations

### Database Persistence
- SQLite database stored in `/data` directory
- Railway provides persistent storage
- Database migrations handled automatically

### Port Management
- Railway uses dynamic port assignment
- No hardcoded ports in application
- Dashboard runs on default Railway port

### Resource Limits
- Monitor memory usage ( trading bot is memory-intensive )
- Consider scaling up if performance issues
- AI API costs still apply

### Logging
- Logs available in Railway dashboard
- Structured logging enabled
- Performance metrics included

## Post-Deployment Checklist

- [ ] Verify bot starts without errors
- [ ] Check database initialization logs
- [ ] Test API key authentication
- [ ] Verify market ingestion starts
- [ ] Confirm dashboard accessibility
- [ ] Set `LIVE_TRADING_ENABLED=false` until verified
- [ ] Monitor resource usage
- [ ] Set up alerts for downtime

## Monitoring

### Railway Dashboard
- View real-time logs
- Monitor resource usage
- Check deployment status

### Application Health
- Bot logs show system status
- Performance metrics every 5 minutes
- Database health checks

### Cost Monitoring
- AI API usage tracked in database
- Daily budget controls active
- Cost alerts in logs

## Troubleshooting

### Common Issues
1. **Import errors**: Check requirements.txt
2. **API failures**: Verify environment variables
3. **Database issues**: Check permissions
4. **Memory limits**: Scale up resources

### Debug Mode
Set `LOG_LEVEL=DEBUG` to see detailed logs.

## Security Notes

- Never commit real API keys to repository
- Use Railway environment variables
- Monitor for unusual trading activity
- Keep `LIVE_TRADING_ENABLED=false` until ready

## Performance Optimization

- Railway provides automatic scaling
- Monitor response times
- Adjust resource allocation as needed
- Consider Redis for caching if needed