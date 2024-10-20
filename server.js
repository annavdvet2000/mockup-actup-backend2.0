require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const OpenAI = require('openai');
const fs = require('fs');
const cors = require('cors');
const similarity = require('compute-cosine-similarity');

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
    
    try {
        const relevantDocs = await findRelevantDocuments(userQuery);
        const context = relevantDocs
            .map(doc => `Interview with ${doc.name}: ${doc.transcript.slice(0, 500)}...`)
            .join("\n");

        const response = await openai.chat.completions.create({
            model: 'gpt-4-turbo',
            messages: [
                { role: 'system', content: 'You are an assistant with access to interview transcripts.' },
                { role: 'user', content: userQuery },
                { role: 'assistant', content: `Here are some relevant excerpts:\n${context}` }
            ],
            max_tokens: 500,
            temperature: 0.7,
        });

        if (response?.choices?.[0]?.message?.content) {
            res.json({ response: response.choices[0].message.content });
        } else {
            res.status(500).json({ error: 'Invalid response format' });
        }
    } catch (error) {
        console.error('Chat Error:', error);
        res.status(500).json({ error: 'Failed to process chat request' });
    }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
