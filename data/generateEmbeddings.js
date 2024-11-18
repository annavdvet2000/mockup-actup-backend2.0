require('dotenv').config();
const fs = require('fs');
const OpenAI = require('openai');

// Initialize OpenAI client using the API key
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

// Load the interview data from metadata.json
const data = JSON.parse(fs.readFileSync('./metadata.json', 'utf-8'));

// Function to generate embeddings for each interview transcript
async function generateEmbeddings() {
    try {
        // Iterate over each interview in the data
        for (const interview of data.interviews) {
            console.log(`Generating embedding for interview ID: ${interview.id}`);
            
            // Get the text content to generate an embedding from
            const text = interview.transcript;
            if (!text || text.trim() === "") {
                console.error(`No transcript found for interview ID: ${interview.id}`);
                continue; // Skip this interview if there's no text to process
            }

            // Generate the embedding using OpenAI's API
            const response = await openai.embeddings.create({
                model: 'text-embedding-ada-002',
                input: text, // Ensure 'input' is provided with the text
            });

            // Store the embedding in the interview object
            interview.embedding = response.data[0].embedding;
        }

        // Write updated data with embeddings back to a new JSON file
        fs.writeFileSync('./metadata_with_embeddings.json', JSON.stringify(data, null, 2));
        console.log('Embeddings generated and saved to metadata_with_embeddings.json.');
    } catch (error) {
        console.error('Error generating embeddings:', error);
    }
}

// Run the embedding generation
generateEmbeddings();