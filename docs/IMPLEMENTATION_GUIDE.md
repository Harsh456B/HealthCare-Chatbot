# Medical-Chatbot - Implementation Guide for Improvements

This guide provides step-by-step instructions to implement the recommended enhancements.

## 1. Security: Add Rate Limiting

### Install dependency
```bash
pip install flask-limiter
```

### Update `app.py`
Add at the top after imports:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
```

Apply to route:
```python
@app.route("/get", methods=["POST"])
@limiter.limit("10 per minute")
def get_response():
    # ... existing code
```

---

## 2. Input Validation & Security

### Update `/get` route in `app.py`
```python
@app.route("/get", methods=["POST"])
@limiter.limit("10 per minute")
def get_response():
    """Handle chat requests with validation"""
    user_message = request.form.get("msg", "").strip()
    
    # Validation checks
    if not user_message:
        return json.dumps({"error": "Empty message"}), 400
    
    if len(user_message) > 500:
        return json.dumps({"error": "Message too long (max 500 chars)"}), 400
    
    # Prevent prompt injection - add basic filtering
    prohibited_patterns = ["DROP TABLE", "DELETE FROM", "UNION SELECT"]
    if any(pattern in user_message.upper() for pattern in prohibited_patterns):
        return json.dumps({"error": "Invalid input"}), 400
    
    try:
        result = rag_pipeline.invoke({"input": user_message})
        bot_response = result.get("answer", "I couldn't generate a response.")
        return str(bot_response)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return "An error occurred. Please try again."
```

---

## 3. Implement Proper Logging

### Create `logging_config.py` in root directory
```python
import logging
import logging.handlers
import os

def setup_logging(log_level=logging.INFO):
    """Configure application logging"""
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger('medical_chatbot')
    logger.setLevel(log_level)
    
    # File handler (rotates daily)
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/medical_chatbot.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()
```

### Update `app.py` to use logging
```python
from logging_config import logger

# Replace all print() statements with logger calls
logger.info(f"Incoming request - Message length: {len(user_message)}")
logger.error(f"Error processing request: {str(e)}")
logger.info("Vector store initialized successfully")
```

---

## 4. Add Security Headers

### Update `app.py`
```python
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

# Enable CORS if needed
from flask_cors import CORS
CORS(app, resources={r"/get": {"origins": ["https://yourdomain.com"]}})
```

---

## 5. Enhanced Error Handling

### Create `error_handlers.py`
```python
from flask import jsonify
import logging

logger = logging.getLogger(__name__)

class CustomException(Exception):
    """Base exception for the application"""
    def __init__(self, message, status_code=500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class InvalidInputError(CustomException):
    status_code = 400

class ServiceError(CustomException):
    status_code = 503

def register_error_handlers(app):
    """Register error handlers"""
    
    @app.errorhandler(InvalidInputError)
    def handle_invalid_input(error):
        logger.warning(f"Invalid input: {error.message}")
        return jsonify({"error": error.message}), error.status_code
    
    @app.errorhandler(ServiceError)
    def handle_service_error(error):
        logger.error(f"Service error: {error.message}")
        return jsonify({"error": error.message}), error.status_code
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        logger.critical(f"Internal server error: {str(error)}")
        return jsonify({"error": "Internal server error"}), 500
```

---

## 6. Frontend Enhancement: Persist Chat History

### Update `templates/chat.html` - Add in `<head>`
```html
<script>
    // Initialize chat history storage
    const STORAGE_KEY = 'medicalChatHistory';
    
    function saveChatHistory(messages) {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
        } catch (e) {
            console.warn('Could not save chat history:', e);
        }
    }
    
    function loadChatHistory() {
        try {
            return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
        } catch (e) {
            console.warn('Could not load chat history:', e);
            return [];
        }
    }
    
    function clearChatHistory() {
        localStorage.removeItem(STORAGE_KEY);
    }
</script>
```

### Update chat submission handler (in existing script)
```javascript
$(document).ready(function() {
    let chatHistory = loadChatHistory();
    
    // Display saved messages on load
    chatHistory.forEach(msg => {
        displayMessage(msg);
    });
    
    // Update to save each message
    chatForm.on("submit", function(event) {
        // ... existing code ...
        
        // After getting response:
        chatHistory.push({
            type: 'user',
            content: userMessage,
            time: timeString
        });
        chatHistory.push({
            type: 'bot',
            content: response,
            time: timeString
        });
        
        saveChatHistory(chatHistory);
    });
});
```

### Add clear history button to UI
```html
<div class="card-header msg_head">
    <div class="d-flex bd-highlight">
        <!-- ... existing code ... -->
        <button onclick="clearChatHistory(); location.reload();" 
                style="position:absolute; right:10px; top:10px; 
                       background: none; border: none; color: white; cursor: pointer;">
            <i class="fas fa-trash"></i>
        </button>
    </div>
</div>
```

---

## 7. Add Response Caching

### Create `cache_manager.py`
```python
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict

class ResponseCache:
    """Simple in-memory cache for responses"""
    
    def __init__(self, ttl_minutes: int = 30):
        self.cache: Dict[str, tuple] = {}
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def _hash_query(self, query: str) -> str:
        """Create hash of query for cache key"""
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[str]:
        """Get cached response if exists and not expired"""
        key = self._hash_query(query)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return response
            else:
                del self.cache[key]
        return None
    
    def set(self, query: str, response: str):
        """Cache a response"""
        key = self._hash_query(query)
        self.cache[key] = (response, datetime.now())
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
```

### Update `app.py` to use cache
```python
from cache_manager import ResponseCache

response_cache = ResponseCache(ttl_minutes=30)

@app.route("/get", methods=["POST"])
@limiter.limit("10 per minute")
def get_response():
    user_message = request.form.get("msg", "").strip()
    
    # Check cache first
    cached_response = response_cache.get(user_message)
    if cached_response:
        logger.info(f"Cache hit for query: {user_message[:50]}")
        return cached_response
    
    try:
        result = rag_pipeline.invoke({"input": user_message})
        bot_response = result.get("answer", "I couldn't generate a response.")
        
        # Cache the response
        response_cache.set(user_message, str(bot_response))
        
        return str(bot_response)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return "An error occurred. Please try again."
```

---

## 8. Add Query Analytics

### Create `analytics.py`
```python
import json
from datetime import datetime
from pathlib import Path

class QueryAnalytics:
    """Track query statistics"""
    
    def __init__(self, log_file: str = "logs/queries.jsonl"):
        self.log_file = log_file
        Path(self.log_file).parent.mkdir(exist_ok=True)
    
    def log_query(self, user_query: str, bot_response: str, response_time: float):
        """Log query and response"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": user_query,
            "query_length": len(user_query),
            "response_length": len(bot_response),
            "response_time_ms": response_time * 1000
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

analytics = QueryAnalytics()
```

### Use in app.py
```python
import time

@app.route("/get", methods=["POST"])
def get_response():
    start_time = time.time()
    
    # ... process query ...
    
    response_time = time.time() - start_time
    analytics.log_query(user_message, bot_response, response_time)
    
    return str(bot_response)
```

---

## 9. Configuration Management Enhancement

### Create `config.py`
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    DEBUG = False
    TESTING = False
    
    # API Keys
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Pinecone
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
    EMBEDDING_DIMENSION = 384
    
    # Groq
    GROQ_MODEL_NAME = "llama-3.1-8b-instant"
    GROQ_TEMPERATURE = 0
    
    # RAG
    RETRIEVAL_K = 3
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 20
    
    # Rate Limiting
    RATE_LIMIT = "10 per minute"
    RATE_LIMIT_DAILY = "200 per day"
    
    # Cache
    CACHE_TTL_MINUTES = 30
    
    # Logging
    LOG_LEVEL = "INFO"

class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    DEBUG = False
    RATE_LIMIT = "5 per minute"
    LOG_LEVEL = "WARNING"

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
```

### Update `app.py` to use config
```python
from config import config

env = os.getenv('FLASK_ENV', 'development')
app.config.from_object(config[env])

PINECONE_INDEX_NAME = app.config['PINECONE_INDEX_NAME']
```

---

## 10. Database Backup Strategy

### Create `backup_manager.py`
```python
import os
import json
from datetime import datetime
from pathlib import Path

class BackupManager:
    """Manage vector database backups"""
    
    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self, index_stats: dict):
        """Create backup metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"backup_{timestamp}.json"
        
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "index_stats": index_stats
        }
        
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        print(f"Backup created: {backup_file}")
    
    def list_backups(self):
        """List all backups"""
        return sorted(self.backup_dir.glob("backup_*.json"))
```

---

## Implementation Priority

**Week 1 (Critical)**:
1. Rate limiting (security)
2. Input validation (security)
3. Logging setup

**Week 2 (Important)**:
4. Security headers
5. Error handling
6. Configuration management

**Week 3 (Enhancement)**:
7. Chat history persistence
8. Response caching
9. Query analytics

**Week 4 (Optional)**:
10. Backup strategy
11. Advanced monitoring
12. Performance optimization

---

## Testing Commands

### Test rate limiting
```bash
for i in {1..15}; do
  curl -X POST http://localhost:8080/get \
    -d "msg=test question $i"
done
```

### Test input validation
```bash
curl -X POST http://localhost:8080/get \
  -d "msg=$(python -c 'print("x" * 600)')"  # 600 chars
```

### Check logs
```bash
tail -f logs/medical_chatbot.log
```

---

## Deployment Checklist

- [ ] Rate limiting installed and configured
- [ ] Input validation implemented
- [ ] Logging configured with rotation
- [ ] Security headers added
- [ ] Error handlers registered
- [ ] Configuration externalized
- [ ] Chat history persistence added
- [ ] Response caching implemented
- [ ] Analytics logging enabled
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Debug mode disabled in production
- [ ] Environment variables configured
- [ ] Load tested (50+ concurrent users)

