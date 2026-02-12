// State management
let isProcessing = false;

// DOM elements
const chatbot = document.getElementById('chatbot');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const statusEl = document.getElementById('status');
const responseTimeEl = document.getElementById('response-time');
const queueCountEl = document.getElementById('queue-count');
const showSourcesCheckbox = document.getElementById('show-sources');
const showThinkingCheckbox = document.getElementById('show-thinking');
const debugCheckbox = document.getElementById('debug');
const topKSlider = document.getElementById('top-k');
const topKValue = document.getElementById('top-k-value');

// Update slider value display
topKSlider.addEventListener('input', (e) => {
    topKValue.textContent = e.target.value;
});

// Handle Enter key (Shift+Enter for new line)
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Send button click
sendBtn.addEventListener('click', sendMessage);

// Use Server-Sent Events for real-time status updates
let eventSource = null;

function connectSSE() {
    if (eventSource) {
        eventSource.close();
    }
    
    eventSource = new EventSource('/api/status-stream');
    
    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            queueCountEl.textContent = data.active_requests;
            // Optionally update other metrics if needed
        } catch (error) {
            console.error('SSE parse error:', error);
        }
    };
    
    eventSource.onerror = function(error) {
        console.error('SSE connection error:', error);
        eventSource.close();
        // Reconnect after 5 seconds
        setTimeout(connectSSE, 5000);
    };
    
    console.log('✅ SSE connection established');
}

// Start SSE connection
connectSSE();

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (eventSource) {
        eventSource.close();
    }
});

// Send message function
async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message || isProcessing) return;
    
    if (message.length > 500) {
        addBotMessage('⚠️ Query too long. Maximum 500 characters.');
        return;
    }
    
    // Clear welcome message if present
    const welcomeMsg = chatbot.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    // Add user message
    addUserMessage(message);
    
    // Clear input
    messageInput.value = '';
    
    // Set processing state
    isProcessing = true;
    sendBtn.disabled = true;
    statusEl.textContent = '⏳ Processing...';
    statusEl.classList.add('processing');
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                show_sources: showSourcesCheckbox.checked,
                show_thinking: showThinkingCheckbox.checked,
                debug: debugCheckbox.checked,
                top_k: parseInt(topKSlider.value)
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addBotMessage(data.answer, data.thinking, data.sources, data.response_time);
            responseTimeEl.textContent = `${data.response_time}s`;
        } else {
            addBotMessage(data.error || 'An error occurred');
        }
        
    } catch (error) {
        console.error('Error:', error);
        addBotMessage('❌ Connection error. Please try again.');
    } finally {
        isProcessing = false;
        sendBtn.disabled = false;
        statusEl.textContent = 'Ready';
        statusEl.classList.remove('processing');
    }
}

// Add user message to chat
function addUserMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.textContent = text;
    chatbot.appendChild(messageDiv);
    scrollToBottom();
}

// Add bot message to chat
function addBotMessage(answer, thinking = null, sources = null, responseTime = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';
    
    // Check if this is a timeout response
    const isTimeout = answer.includes('Model hallucinated, please rephrase your question');
    if (isTimeout) {
        messageDiv.classList.add('timeout-message');
    }
    
    let html = '';
    
    // Add thinking block if present
    if (thinking) {
        // Render thinking as markdown
        const thinkingHtml = marked.parse(thinking);
        html += `<div class="thinking-block">💭 <strong>THINKING:</strong><div class="thinking-content">${thinkingHtml}</div></div>`;
    }
    
    // Parse and render answer as markdown
    const answerHtml = marked.parse(answer);
    html += `<div class="answer-content">${answerHtml}</div>`;
    
    // Add sources if present
    if (sources && sources.length > 0) {
        html += '<div class="sources-block">';
        html += '<div class="sources-title">📚 SOURCES:</div>';
        sources.forEach(source => {
            html += `<div class="source-item">${source.index}. ${escapeHtml(source.filename)}</div>`;
        });
        html += '</div>';
    }
    
    // Add response time
    if (responseTime) {
        const timeClass = responseTime >= 90 ? 'timeout-indicator' : '';
        html += `<div class="response-time ${timeClass}">⏱️ ${responseTime}s</div>`;
    }
    
    messageDiv.innerHTML = html;
    chatbot.appendChild(messageDiv);
    scrollToBottom();
}

// Legacy polling function (kept as fallback if SSE fails)
async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        queueCountEl.textContent = data.active_requests;
    } catch (error) {
        console.error('Status update error:', error);
    }
}

// Scroll chat to bottom
function scrollToBottom() {
    chatbot.scrollTop = chatbot.scrollHeight;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});
