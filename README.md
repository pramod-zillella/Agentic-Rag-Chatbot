# Agentic RAG Fitness Chatbot

## Overview
The Agentic RAG Fitness Chatbot is an AI-powered application designed to provide personalized fitness guidance and workout recommendations. Uses Retrieval-Augmented Generation (RAG), multi-agent systems, and a curated knowledge base, to deliver context-aware and actionable fitness advice. The chatbot focuses on empowering beginners with evidence-based fitness recommendations, real-time video demonstrations, and safety guidelines.

### LangGraph Workflow
The LangGraph Workflow outlines the sequence of nodes in the system’s architecture. Each node represents a specific function, from query handling to generating responses. Below is the compiled workflow:

![LangGraph Workflow](https://github.com/pramod-zillella/Agentic-Rag-Chatbot/blob/main/LangGraph-Workflow.png)

## Architecture
The system consists of the following components:

1. **User Input**: Accepts user queries through a Streamlit-based chat interface.
2. **Query Refinement**: Transforms user input into an optimized format using an LLM-based query rewriting mechanism.
3. **Retrieval System**:
   - Encodes user queries using Sentence Transformers.
   - Queries the Pinecone vector database to fetch relevant fitness data.
   - Retrieves video demonstrations and transcripts.
4. **Response Generation**: Synthesizes retrieved information using GPT-4o, ensuring the response is actionable and grounded in context.
5. **Video Recommendations**: Displays video thumbnails, titles, and links alongside detailed transcripts.
6. **Langsmith Integration**: Tracks agent-level decisions and improves overall system reliability.

## Langsmith Trace and LangGraph Workflow
To provide transparency and insights into the system's behavior, the Langsmith trace and LangGraph workflow have been visualized:

### Langsmith Trace
The Langsmith Trace captures the flow of the chatbot’s decision-making process, including tool calls and their respective responses. Below is an example trace showcasing a user query and the system's response:

![Langsmith Trace](https://github.com/pramod-zillella/AgenticRagChatbot/blob/main/Langsmith-Trace.png)

## Installation

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/pramod-zillella/AgenticRagChatbot.git
   cd agentic-rag-fitness-chatbot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file in the project directory.
   - Add your API keys:
     ```
     OPENAI_API_KEY=your_openai_api_key
     PINECONE_API_KEY=your_pinecone_api_key
     LANGCHAIN_API_KEY_V2=your_langchain_api_key
     ```
4. Run the Streamlit application:
   ```bash
   streamlit run interface.py
   ```

## Usage
- **Predefined Questions**: Select from common fitness-related queries or type your own.
- **Custom Queries**: Ask personalized questions about workouts, nutrition, or injury prevention.
- **Interactive Recommendations**: View suggested video demonstrations and detailed response within the chat interface.

