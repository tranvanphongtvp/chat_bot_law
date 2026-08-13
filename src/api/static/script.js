document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatMessages = document.getElementById('chatMessages');
    const sendBtn = document.getElementById('sendBtn');
    const modelSelect = document.getElementById('modelSelect');
    const newChatBtn = document.getElementById('newChatBtn');
    const llmStatus = document.getElementById('llmStatus');

    // --- Auto-fetch danh sách model từ API ---
    async function fetchLLMStatus() {
        try {
            const res = await fetch('/api/llm-status');
            const data = await res.json();

            if (data.status === 'running' && data.models.length > 0) {
                // Xóa option "Đang tải..."
                modelSelect.innerHTML = '';
                data.models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m;
                    opt.textContent = m;
                    modelSelect.appendChild(opt);
                });
                llmStatus.innerHTML = '🟢 Sẵn sàng (Groq)';
                llmStatus.className = 'status-badge online';
            } else {
                modelSelect.innerHTML = '<option value="" disabled selected>❌ Thiếu API Key</option>';
                llmStatus.innerHTML = '🔴 Lỗi cấu hình';
                llmStatus.className = 'status-badge offline';
            }
        } catch (e) {
            modelSelect.innerHTML = '<option value="" disabled selected>❌ Lỗi kết nối</option>';
            llmStatus.innerHTML = '🔴 Không kết nối';
            llmStatus.className = 'status-badge offline';
        }
    }

    // Gọi ngay khi load trang
    fetchLLMStatus();

    // Tạo phần tử Loading
    function createLoadingIndicator() {
        const div = document.createElement('div');
        div.className = 'message bot loading-msg';
        div.innerHTML = `
            <div class="avatar"><i class="fa-solid fa-scale-balanced"></i></div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        return div;
    }

    // Format text với xuống dòng
    function formatText(text) {
        return text.split('\n').map(line => `<p>${line}</p>`).join('');
    }

    // Thêm tin nhắn vào giao diện
    function appendMessage(role, text, citations = null) {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        
        let contentHtml = `<div class="message-content">${formatText(text)}`;
        
        // Thêm citation nếu có
        if (citations && citations.length > 0) {
            const template = document.getElementById('citationTemplate').content.cloneNode(true);
            const ul = template.querySelector('.citation-list');
            
            citations.forEach(c => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <div class="cite-title">${c.law} - ${c.article}</div>
                    <div class="cite-text">"${c.content}"</div>
                `;
                ul.appendChild(li);
            });
            
            const citationContainer = document.createElement('div');
            citationContainer.appendChild(template);
            contentHtml += citationContainer.innerHTML;
        }
        
        contentHtml += `</div>`;
        
        let avatarIcon = role === 'user' ? 'fa-user' : 'fa-scale-balanced';
        div.innerHTML = `
            <div class="avatar"><i class="fa-solid ${avatarIcon}"></i></div>
            ${contentHtml}
        `;
        
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Xử lý submit
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const text = userInput.value.trim();
        if (!text) return;
        
        // Disable input
        userInput.value = '';
        userInput.disabled = true;
        sendBtn.disabled = true;
        
        // Hiển thị user message
        appendMessage('user', text);
        
        // Hiển thị loading
        const loadingMsg = createLoadingIndicator();
        chatMessages.appendChild(loadingMsg);
        scrollToBottom();
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: text,
                    model: modelSelect.value
                })
            });
            
            // Xóa loading
            loadingMsg.remove();
            
            if (!response.ok) {
                throw new Error('Lỗi từ server: ' + response.statusText);
            }
            
            const data = await response.json();
            
            if (data.error) {
                appendMessage('bot', data.answer, data.citations);
                console.error(data.error);
            } else {
                appendMessage('bot', data.answer, data.citations);
            }
            
        } catch (error) {
            loadingMsg.remove();
            appendMessage('bot', `Xin lỗi, có lỗi xảy ra: ${error.message}`);
        } finally {
            // Enable input
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    });

    newChatBtn.addEventListener('click', () => {
        chatMessages.innerHTML = `
            <div class="message bot">
                <div class="avatar"><i class="fa-solid fa-scale-balanced"></i></div>
                <div class="message-content">
                    <p>Phiên trò chuyện đã được làm mới. Bạn có câu hỏi nào khác không?</p>
                </div>
            </div>
        `;
    });
});
