// Configuration
// 6e9e80e9fd0e424d8a5cb3ac8ff820e9
const CONFIG = {
    threadId: '6',
    ownerId: 'e1e6b21c463b4a1f9b5ac4c0255f9e27',
    apiEndpoint: '/api/chat'
};

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const downloadSection = document.getElementById('downloadSection');
const downloadContent = document.getElementById('downloadContent');
const vizContainer = document.getElementById('visualization-container');

// Add user message to chat
function addUserMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `
        <div class="message-content">${escapeHtml(text)}</div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Add or update assistant message
let currentAssistantMessage = null;

function updateAssistantMessage(content) {
    if (!currentAssistantMessage) {
        currentAssistantMessage = document.createElement('div');
        currentAssistantMessage.className = 'message assistant';
        currentAssistantMessage.innerHTML = `
            <div class="message-content"></div>
        `;
        chatMessages.appendChild(currentAssistantMessage);
    }

    const contentDiv = currentAssistantMessage.querySelector('.message-content');
    
    // Parse markdown
    try {
        contentDiv.innerHTML = marked.parse(content);
    } catch (e) {
        console.error('Markdown parse error:', e);
        contentDiv.textContent = content;
    }
    
    scrollToBottom();
}

function finalizeAssistantMessage() {
    currentAssistantMessage = null;
}

// Show loading in download section
function showLoading(mongodbContent, loadingHTML) {
    downloadSection.classList.add('active');
    downloadContent.innerHTML = loadingHTML;
    updateAssistantMessage(mongodbContent);
}

// Show visualization
function showVisualization(figData) {
    if (figData) {
        vizContainer.innerHTML = '';
        Plotly.newPlot('visualization-container', figData.data, figData.layout);
        
        // Add download button
        downloadContent.innerHTML = `
            <p>กราฟถูกสร้างเรียบร้อยแล้ว</p>
            <button class="download-button" onclick="downloadGraph()">
                ดาวน์โหลดกราฟ (PNG)
            </button>
            <button class="download-button" onclick="downloadGraphHTML()">
                ดาวน์โหลดกราฟ (HTML)
            </button>
        `;
    }
}

// Download graph as PNG
function downloadGraph() {
    Plotly.downloadImage('visualization-container', {
        format: 'png',
        width: 1200,
        height: 800,
        filename: 'graph'
    });
}

// Download graph as HTML
function downloadGraphHTML() {
    const graphDiv = document.getElementById('visualization-container');
    Plotly.toImage(graphDiv, {format: 'svg'});
}

// Scroll to bottom
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Handle form submission
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const message = userInput.value.trim();
    if (!message) return;

    // Add user message
    addUserMessage(message);
    userInput.value = '';

    // Disable input
    userInput.disabled = true;
    const sendButton = chatForm.querySelector('.send-button');
    sendButton.disabled = true;

    try {
        // Call API
        const response = await fetch(CONFIG.apiEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_message: message,
                thread_id: CONFIG.threadId,
                owner_id: CONFIG.ownerId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = ''; // เพิ่ม buffer สำหรับเก็บ chunk ที่ยังไม่สมบูรณ์

        while (true) {
            const {value, done} = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, {stream: true});
            const lines = buffer.split('\n');
            
            // เก็บบรรทัดสุดท้ายไว้ใน buffer กรณียังไม่สมบูรณ์
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const jsonStr = line.slice(6).trim();
                        if (jsonStr) {
                            const data = JSON.parse(jsonStr);
                            handleStreamData(data);
                        }
                    } catch (e) {
                        console.error('JSON parse error:', e, 'Line:', line);
                    }
                }
            }
        }

        // ประมวลผล buffer ที่เหลือ
        if (buffer.trim() && buffer.startsWith('data: ')) {
            try {
                const jsonStr = buffer.slice(6).trim();
                if (jsonStr) {
                    const data = JSON.parse(jsonStr);
                    handleStreamData(data);
                }
            } catch (e) {
                console.error('JSON parse error (buffer):', e);
            }
        }

        finalizeAssistantMessage();

    } catch (error) {
        console.error('Error:', error);
        updateAssistantMessage(`❌ เกิดข้อผิดพลาด: ${error.message}`);
        finalizeAssistantMessage();
    } finally {
        // Enable input
        userInput.disabled = false;
        sendButton.disabled = false;
        userInput.focus();
    }
});

// Handle streamed data
function handleStreamData(data) {
    console.log('Received data:', data); // Debug log

    // ตรวจสอบว่ามี error หรือไม่
    // if (data.error) {
    //     updateAssistantMessage(`❌ เกิดข้อผิดพลาด: ${data.error}`);
    //     return;
    // }

    // ตรวจสอบ supervisor_output
    if (data.supervisor_output !== undefined) {
        updateAssistantMessage(data.supervisor_output);
    }
    // ตรวจสอบ mongodb_output
    else if (data.mongodb_output !== undefined) {
        updateAssistantMessage(data.mongodb_output);
    }
    // ตรวจสอบ loading_code
    else if (data.loading_code !== undefined) {
        if (Array.isArray(data.loading_code) && data.loading_code.length >= 2) {
            showLoading(data.loading_code[0], data.loading_code[1]);
        }
    }
    // ตรวจสอบ final_output
    else if (data.final_output !== undefined) {
        const final = data.final_output;
        
        // Update final message
        if (final.messages && Array.isArray(final.messages) && final.messages.length > 0) {
            const lastMessage = final.messages[final.messages.length - 1];
            if (lastMessage && lastMessage.content) {
                updateAssistantMessage(lastMessage.content);
            }
        }

        // Show visualization
        if (final.html_fig) {
            showVisualization(final.html_fig);
        }
    }
}

// Initial focus
userInput.focus();