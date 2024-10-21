require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const OpenAI = require('openai');
const fs = require('fs');
const cors = require('cors');
const similarity = require('compute-cosine-similarity');
const path = require('path');
const crypto = require('crypto');

const app = express();
app.use(bodyParser.json());

app.use(cors({
    origin: ['http://localhost:10000', 'https://mockup-actup.netlify.app'],
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type']
}));

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

// Chat Logger Class
class ChatLogger {
    constructor() {
        this.logPath = path.join(__dirname, 'logs', 'conversations.json');
        this.ensureLogFile();
    }

    ensureLogFile() {
        if (!fs.existsSync(path.dirname(this.logPath))) {
            fs.mkdirSync(path.dirname(this.logPath), { recursive: true });
        }
        if (!fs.existsSync(this.logPath)) {
            fs.writeFileSync(this.logPath, JSON.stringify({ sessions: [] }, null, 2));
        }
    }

    getSession(sessionId) {
        const logs = JSON.parse(fs.readFileSync(this.logPath, 'utf8'));
        return logs.sessions.find(s => s.sessionId === sessionId);
    }

    createSession() {
        const sessionId = 'session_' + crypto.randomBytes(16).toString('hex');
        const logs = JSON.parse(fs.readFileSync(this.logPath, 'utf8'));
        const newSession = {
            sessionId,
            startTime: new Date().toISOString(),
            lastActive: new Date().toISOString(),
            conversations: []
        };
        logs.sessions.push(newSession);
        fs.writeFileSync(this.logPath, JSON.stringify(logs, null, 2));
        return sessionId;
    }

    logConversation(sessionId, userMessage, botResponse, relevantDocs) {
        try {
            const logs = JSON.parse(fs.readFileSync(this.logPath, 'utf8'));
            let session = logs.sessions.find(s => s.sessionId === sessionId);
            
            if (!session) {
                session = {
                    sessionId,
                    startTime: new Date().toISOString(),
                    conversations: []
                };
                logs.sessions.push(session);
            }

            session.lastActive = new Date().toISOString();
            session.conversations.push({
                id: Date.now().toString(),
                timestamp: new Date().toISOString(),
                userMessage,
                botResponse,
                relevantDocuments: relevantDocs.map(doc => doc.id)
            });

            fs.writeFileSync(this.logPath, JSON.stringify(logs, null, 2));
            console.log(`Conversation logged for session ${sessionId}`);
        } catch (error) {
            console.error('Error logging conversation:', error);
        }
    }

    getAllSessions() {
        try {
            const logs = JSON.parse(fs.readFileSync(this.logPath, 'utf8'));
            return logs.sessions;
        } catch (error) {
            console.error('Error reading sessions:', error);
            return [];
        }
    }
}

const chatLogger = new ChatLogger();

// Load metadata with transcripts and embeddings
let data;
try {
    data = JSON.parse(fs.readFileSync('./metadata_with_embeddings.json', 'utf-8'));
} catch (error) {
    console.error('Error loading metadata:', error);
}

app.get('/metadata', (req, res) => {
    if (data) {
        res.json(data);
    } else {
        res.status(500).json({ error: 'Metadata not available' });
    }
});

app.post('/search', (req, res) => {
    const query = req.body.query.toLowerCase();
    const filteredInterviews = data.interviews.filter(interview =>
        interview.name.toLowerCase().includes(query) ||
        interview.excerpt_title.toLowerCase().includes(query) ||
        interview.tags.some(tag => tag.toLowerCase().includes(query))
    );
    res.json({ interviews: filteredInterviews });
});

async function getEmbedding(text) {
    const response = await openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: text,
    });
    return response.data[0].embedding;
}

async function findRelevantDocuments(query) {
    const queryEmbedding = await getEmbedding(query);
    return data.interviews
        .map(doc => ({
            ...doc,
            similarity: similarity(queryEmbedding, doc.embedding),
        }))
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 3);
}

app.post('/chat', async (req, res) => {
    const userQuery = req.body.message;
    const sessionId = req.body.sessionId || chatLogger.createSession();
    
    try {
        const relevantDocs = await findRelevantDocuments(userQuery);
        const context = relevantDocs
            .map(doc => `Interview with ${doc.name}: ${doc.transcript}`)
            .join("\n");

        const response = await openai.chat.completions.create({
            model: 'gpt-4-turbo',
            messages: [
                { 
                    role: 'system', 
                    content: 'You are an assistant with access to the ACT UP Oral History Project interviews. You can access all interview metadata and transcripts to answer questions accurately. Always base your answers on the provided interview data.' 
                },
                { role: 'user', content: userQuery },
                { role: 'assistant', content: `Here are some relevant excerpts:\n${context}` }
            ],
            max_tokens: 500,
            temperature: 0.7,
        });

        const botResponse = response.choices[0].message.content;
        
        // Log the conversation
        chatLogger.logConversation(sessionId, userQuery, botResponse, relevantDocs);

        res.json({ 
            response: botResponse,
            sessionId: sessionId
        });
    } catch (error) {
        console.error('Chat Error:', error);
        res.status(500).json({ error: 'Failed to process chat request' });
    }
});

// Add endpoints to manage and view sessions
app.get('/chat-logs/sessions', (req, res) => {
    const sessions = chatLogger.getAllSessions();
    res.json({ sessions });
});

app.get('/chat-logs/session/:sessionId', (req, res) => {
    const session = chatLogger.getSession(req.params.sessionId);
    if (session) {
        res.json(session);
    } else {
        res.status(404).json({ error: 'Session not found' });
    }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
