require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const OpenAI = require('openai');
const fs = require('fs');
const cors = require('cors');
const similarity = require('compute-cosine-similarity');
const path = require('path');

// Decode and save the service account key
const encodedKey = process.env.GOOGLE_APPLICATION_CREDENTIALS_BASE64;
const serviceKeyPath = path.resolve(__dirname, 'service-account.json');
fs.writeFileSync(serviceKeyPath, Buffer.from(encodedKey, 'base64').toString('utf-8'));

// Point to the service key
process.env.GOOGLE_APPLICATION_CREDENTIALS = serviceKeyPath;

// Load the Google Cloud credentials for Dialogflow
const { SessionsClient } = require('@google-cloud/dialogflow'); // Dialogflow sessions client

const app = express();
app.use(bodyParser.json());
app.use(cors()); // Enable CORS

// Initialize OpenAI client using the API key
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

// Load metadata with transcripts and embeddings
let data;
try {
    data = JSON.parse(fs.readFileSync('./metadata_with_embeddings.json', 'utf-8'));
} catch (error) {
    console.error('Error loading metadata:', error);
}

// Metadata endpoint
app.get('/metadata', (req, res) => {
    if (data) {
        res.json(data); // Sends the metadata
    } else {
        res.status(500).json({ error: 'Metadata not available' });
    }
});

// Search endpoint
app.post('/search', (req, res) => {
    const query = req.body.query.toLowerCase();
    const filteredInterviews = data.interviews.filter(interview =>
        interview.name.toLowerCase().includes(query) ||
        interview.excerpt_title.toLowerCase().includes(query) ||
        interview.tags.some(tag => tag.toLowerCase().includes(query))
    );
    res.json({ interviews: filteredInterviews });
});

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

    // Log the incoming request for debugging
    console.log('Received Dialogflow webhook request:', req.body);

    try {
        const relevantDocs = await findRelevantDocuments(userQuery);
        const context = relevantDocs.map(doc =>
            `Interview with ${doc.name}: ${doc.transcript.slice(0, 500)}...`
        ).join("\n");

        // Log the context created from relevant documents
        console.log('Generated context for ChatGPT:', context);

        // Make the API request to OpenAI
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

        // Log the full API response for debugging
        console.log('ChatGPT API raw response:', JSON.stringify(response, null, 2));

        // Check if 'choices' exists and is not empty
        if (response && response.data && response.data.choices && response.data.choices.length > 0) {
            const chatgptResponse = response.data.choices[0].message.content;
            res.json({
                fulfillmentText: chatgptResponse
            });
        } else {
            // Log if choices is missing or response is invalid
            console.error('No choices found in the ChatGPT API response:', response.data);
            res.status(500).json({
                fulfillmentText: "Sorry, I couldn't understand the response from ChatGPT."
            });
        }
    } catch (error) {
        // Enhanced error logging for failed API requests
        if (error.response) {
            console.error('Error communicating with ChatGPT - Status Code:', error.response.status);
            console.error('Error data:', JSON.stringify(error.response.data, null, 2));
        } else {
            console.error('Error communicating with ChatGPT:', error.message);
        }
        res.json({
            fulfillmentText: 'Sorry, something went wrong while processing your request.'
        });
    }
});

// Test OpenAI API locally
app.post('/test-openai', async (req, res) => {
    const userQuery = req.body.query;
    try {
        const response = await openai.chat.completions.create({
            model: 'gpt-4-turbo',
            messages: [
                { role: 'system', content: 'You are a helpful assistant.' },
                { role: 'user', content: userQuery }
            ],
            max_tokens: 500,
            temperature: 0.7,
        });

        if (response && response.data && response.data.choices) {
            const chatgptResponse = response.data.choices[0].message.content;
            res.json({ response: chatgptResponse });
        } else {
            console.error('Unexpected response format from OpenAI:', response.data);
            res.json({ error: 'Unexpected response format' });
        }
    } catch (error) {
        console.error('OpenAI Error:', error.response ? error.response.data : error.message);
        res.status(500).json({ error: 'OpenAI API Error' });
    }
});

// Start the server
const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
