require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const OpenAI = require('openai'); // Correct import for newer versions
const fs = require('fs');
const similarity = require('compute-cosine-similarity');

const app = express();
app.use(bodyParser.json());

// Initialize OpenAI client using the API key
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

// Load metadata with transcripts and embeddings
let data = JSON.parse(fs.readFileSync('./metadata_with_embeddings.json', 'utf-8'));

// Function to get embeddings for a given text
async function getEmbedding(text) {
    const response = await openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: text,
    });
    return response.data[0].embedding;
}

// Function to find the most relevant documents based on a query
async function findRelevantDocuments(query) {
    const queryEmbedding = await getEmbedding(query);
    const results = data.interviews.map(doc => ({
        ...doc,
        similarity: similarity(queryEmbedding, doc.embedding),
    }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 3);
    return results;
}

// Dialogflow webhook endpoint
app.post('/webhook', async (req, res) => {
    const userQuery = req.body.queryResult.queryText;
    try {
        const relevantDocs = await findRelevantDocuments(userQuery);
        const context = relevantDocs.map(doc =>
            `Interview with ${doc.name}: ${doc.transcript.slice(0, 500)}...`
        ).join("\n");

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

        const chatgptResponse = response.data.choices[0].message.content;
        res.json({
            fulfillmentText: chatgptResponse
        });
    } catch (error) {
        console.error('Error communicating with ChatGPT:', error.response ? error.response.data : error.message);
        res.json({
            fulfillmentText: 'Sorry, something went wrong while processing your request.'
        });
    }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});