// server.js
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fs = require('fs').promises;
const path = require('path');
const OpenAI = require('openai');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// OpenAI configuration
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

// File paths
const DATA_DIR = path.join(__dirname, 'data');
const METADATA_WITH_EMBEDDINGS_PATH = path.join(DATA_DIR, 'metadata_with_embeddings.json');
const CHAT_LOGS_PATH = path.join(DATA_DIR, 'chat_logs.json');

// Initialize data directory and chat logs
async function initializeData() {
    try {
        await fs.mkdir(DATA_DIR, { recursive: true });
        try {
            await fs.access(CHAT_LOGS_PATH);
        } catch {
            await fs.writeFile(CHAT_LOGS_PATH, JSON.stringify([]));
        }
    } catch (error) {
        console.error('Error initializing data:', error);
    }
}

// Helper function for cosine similarity
function calculateCosineSimilarity(vecA, vecB) {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
    return dotProduct / (magnitudeA * magnitudeB);
}

// Add a test route
app.get('/', (req, res) => {
    res.json({ message: "Server is running" });
});

// Unified OpenAI Endpoint
app.post('/api/openai', async (req, res) => {
    try {
        const { message, sessionId } = req.body;
        console.log('Received message:', message); // Debug log

        // Step 1: Generate an embedding for the user's message
        const messageEmbeddingResponse = await openai.embeddings.create({
            model: "text-embedding-ada-002",
            input: message
        });
        const messageEmbedding = messageEmbeddingResponse.data[0].embedding;

        // Step 2: Load interview embeddings and calculate cosine similarity
        let metadataWithEmbeddings;
        try {
            const data = await fs.readFile(METADATA_WITH_EMBEDDINGS_PATH, 'utf8');
            metadataWithEmbeddings = JSON.parse(data);
        } catch (error) {
            console.error('Error reading metadata file:', error);
            // Provide a fallback if file doesn't exist or is invalid
            metadataWithEmbeddings = { interviews: [] };
        }

        const relevantInterviews = metadataWithEmbeddings.interviews
            .map(interview => {
                const similarity = calculateCosineSimilarity(messageEmbedding, interview.embedding);
                return {
                    text: interview.transcript,
                    similarity,
                    name: interview.name,
                    id: interview.id,
                    tags: interview.tags
                };
            })
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, 3); // Get top 3 relevant interviews

        // Step 3: Create a context string for GPT-4
        const contextString = relevantInterviews
            .map(context => `From ${context.name}'s interview (${context.id}): ${context.text}`)
            .join('\n\n');

        // Step 4: Call GPT-4 with the user message and interview context
        const completion = await openai.chat.completions.create({
            model: "gpt-4-turbo-preview",
            messages: [
                {
                    "role": "system",
                    "content": `You are a helpful assistant for the ACT UP Oral History Project. 
                    Use the following interview context to answer questions:\n${contextString}`
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
        });

        // Step 5: Log the conversation
        try {
            const chatLogs = JSON.parse(await fs.readFile(CHAT_LOGS_PATH, 'utf8') || '[]');
            chatLogs.push({
                sessionId,
                timestamp: new Date().toISOString(),
                message,
                response: completion.choices[0].message.content,
                relevantInterviews: relevantInterviews.map(c => ({ id: c.id, name: c.name }))
            });
            await fs.writeFile(CHAT_LOGS_PATH, JSON.stringify(chatLogs, null, 2));
        } catch (error) {
            console.error('Error logging chat:', error);
            // Continue even if logging fails
        }

        // Step 6: Send the response back to the frontend
        res.json({
            response: completion.choices[0].message.content,
            relevantInterviews: relevantInterviews.map(c => ({
                id: c.id,
                name: c.name,
                tags: c.tags,
                similarity: c.similarity
            }))
        });
    } catch (error) {
        console.error('Error processing request:', error);
        res.status(500).json({ 
            error: 'Failed to process your request.',
            details: error.message 
        });
    }
});

// Initialize and start server
initializeData().then(() => {
    app.listen(PORT, () => {
        console.log(`Server running on port ${PORT}`);
    });
});