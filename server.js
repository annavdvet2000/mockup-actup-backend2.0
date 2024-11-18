require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fs = require('fs').promises;
const path = require('path');
const OpenAI = require('openai');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors({
    origin: ['https://mockup-actup.netlify.app', 'http://localhost:3000'],
    methods: ['GET', 'POST'],
    credentials: true,
    optionsSuccessStatus: 200
}));

app.use(express.json());

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

const DATA_DIR = path.join(__dirname, 'data');
const METADATA_WITH_EMBEDDINGS_PATH = path.join(DATA_DIR, 'metadata_with_embeddings.json');
const CHAT_LOGS_PATH = path.join(DATA_DIR, 'chat_logs.json');

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

function calculateCosineSimilarity(vecA, vecB) {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
    return dotProduct / (magnitudeA * magnitudeB);
}

app.get('/', (req, res) => {
    res.json({ message: "Server is running" });
});

function splitResponseIntoChunks(response, maxTokens) {
    const words = response.split(' ');
    const chunks = [];
    let currentChunk = '';

    for (const word of words) {
        if (currentChunk.length + word.length + 1 <= maxTokens) {
            currentChunk += ' ' + word;
        } else {
            chunks.push(currentChunk.trim());
            currentChunk = word;
        }
    }

    if (currentChunk.trim().length > 0) {
        chunks.push(currentChunk.trim());
    }

    return chunks;
}

app.post('/api/openai', async (req, res) => {
    try {
        const { message, sessionId, max_tokens } = req.body;

        const messageEmbeddingResponse = await openai.embeddings.create({
            model: "text-embedding-ada-002",
            input: message
        });
        const messageEmbedding = messageEmbeddingResponse.data[0].embedding;

        const metadataWithEmbeddings = JSON.parse(await fs.readFile(METADATA_WITH_EMBEDDINGS_PATH, 'utf8'));

        const relevantChunks = metadataWithEmbeddings.interviews.flatMap(interview => 
            interview.chunks.map(chunk => ({
                text: chunk.text,
                similarity: calculateCosineSimilarity(messageEmbedding, chunk.embedding),
                interviewName: interview.name,
                interviewId: interview.id
            }))
        )
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 3);

        const contextString = relevantChunks
            .map(chunk => `From ${chunk.interviewName}'s interview (${chunk.interviewId}): ${chunk.text}`)
            .join('\n\n');

        const completion = await openai.chat.completions.create({
            model: "gpt-4-turbo-preview",
            max_tokens: 2048,
            messages: [
                {
                    "role": "system",
                    "content": `You are a helpful assistant for the ACT UP Oral History Project. 
                    Provide a concise response within the given token limit, using the following interview context:\n${contextString}`
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
        });

        const responseChunks = splitResponseIntoChunks(completion.choices[0].message.content, max_tokens);
        
        try {
            const chatLogs = JSON.parse(await fs.readFile(CHAT_LOGS_PATH, 'utf8') || '[]');
            chatLogs.push({
                sessionId,
                timestamp: new Date().toISOString(),
                message,
                response: responseChunks.join(' ')
            });
            await fs.writeFile(CHAT_LOGS_PATH, JSON.stringify(chatLogs, null, 2));
        } catch (error) {
            console.error('Error logging chat:', error);
        }

        res.json({ response: responseChunks.join(' ') });
    } catch (error) {
        console.error('Error processing request:', error);
        res.status(500).json({ error: 'Failed to process your request', details: error.message });
    }
});

initializeData().then(() => {
    app.listen(PORT, () => {
        console.log(`Server running on port ${PORT}`);
    });
});