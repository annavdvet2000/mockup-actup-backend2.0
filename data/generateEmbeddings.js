require('dotenv').config();
const fs = require('fs').promises;
const path = require('path');
const pdf = require('pdf-parse');
const OpenAI = require('openai');

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

async function readPDF(pdfPath) {
    const dataBuffer = await fs.readFile(pdfPath);
    const data = await pdf(dataBuffer);
    return data.text;
}

function chunkText(text, maxChunkSize = 1000) {
    const sentences = text.split(/[.!?]+\s/);
    const chunks = [];
    let currentChunk = '';

    for (const sentence of sentences) {
        if ((currentChunk + sentence).length > maxChunkSize) {
            if (currentChunk) chunks.push(currentChunk.trim());
            currentChunk = sentence;
        } else {
            currentChunk += ' ' + sentence;
        }
    }
    if (currentChunk) chunks.push(currentChunk.trim());
    return chunks;
}

async function generateEmbeddings(chunks) {
    const embeddings = [];
    for (const chunk of chunks) {
        try {
            const response = await openai.embeddings.create({
                model: "text-embedding-ada-002",
                input: chunk
            });
            embeddings.push({
                text: chunk,
                embedding: response.data[0].embedding
            });
        } catch (error) {
            console.error('Error generating embedding:', error);
        }
    }
    return embeddings;
}

async function main() {
    try {
        const metadata = JSON.parse(await fs.readFile('./metadata.json', 'utf8'));
        const updatedInterviews = [];

        for (const interview of metadata.interviews) {
            console.log(`Processing interview ${interview.id}`);
            
            const pdfPath = path.join(__dirname, interview.transcript_pdf);
            
            try {
                const fullText = await readPDF(pdfPath);
                const chunks = chunkText(fullText);
                const chunkEmbeddings = await generateEmbeddings(chunks);
                
                updatedInterviews.push({
                    ...interview,
                    chunks: chunkEmbeddings
                });
                
            } catch (error) {
                console.error(`Error processing PDF for interview ${interview.id}:`, error);
            }
        }

        await fs.writeFile(
            './metadata_with_embeddings.json', 
            JSON.stringify({ interviews: updatedInterviews }, null, 2)
        );

        console.log('Processing complete');
    } catch (error) {
        console.error('Error in main process:', error);
    }
}

main();