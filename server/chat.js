// import modules
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { PromptTemplate } from "langchain/prompts";

import { PDFLoader } from "langchain/document_loaders/fs/pdf";


// Diverge implementation #1: user prompt should be retrieved
// not only through local pdf file but also online prompt entrance

const chat = async(query, filePath = "./uploads/human.pdf") => {
// load data
const loader = new PDFLoader(filePath);
const data = await loader.load();

// split data (we don't need such large chunk size)
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100,
    chunkOverlap: 0,
});
const splitDocs = await textSplitter.splitDocuments(data);
// create embedding
const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.REACT_APP_OPENAI_API_KEY,
});

const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
);

// retrieve answer
const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    openAIApiKey: process.env.REACT_APP_OPENAI_API_KEY,
});

const template = `Given the following pieces of context or question, 
please provide a personalized daily schedule for the user to follow, using the following format.
If both the context and the question are missing, provide an empty schedule. 
If the context is empty, use the question alone to provide a personalized daily schedule for the user to follow.
If the context is not empty, use the context alone to provide a personalized daily schedule for the user to follow.

{context}
Question: {question}
Here is your daily schedule:
Hour-Minute: Task1
Hour-Minute: Task2
...
Hour-Minute: Taskn`;

const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
    prompt: PromptTemplate.fromTemplate(template),
});

const response = await chain.call({
    query, 
});

return response;
};

export default chat;