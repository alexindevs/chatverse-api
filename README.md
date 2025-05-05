# AI Chat Platform

## Project Overview

People thrive on meaningful conversations. However, access to these interactions can be limited when isolated, whether in remote locations or confined spaces. This project aims to build and deploy an AI-driven platform that allows users to engage in text-based conversations with AI-generated characters.

## Features

### Mandatory Features

- **Text-Based Interaction**: Users can select an AI character and engage in a text-based conversation.
- **Deployment**: The platform is deployed and accessible via a live URL.
- **Documentation**: This README serves as the required documentation, providing an overview of the approach, design decisions, challenges, and key features.

### Bonus Features

- **Dynamic Character Determination**: AI characters can be generated beyond a predefined set.
- **Persistent Conversation History**: Users' conversation history is saved and persists across sessions.
- **Vector Database for Context Retrieval**: A vector-based database is implemented to improve AI responses by retrieving relevant past messages.

## Live Deployment

The project is deployed and accessible at:
[AI Characters Platform](https://ai-fantasy.onrender.com/)

API Documentation: [Swagger UI](https://ai-fantasy.onrender.com/docs)

## Technical Implementation

### Backend

- **Framework**: FastAPI
- **Database**: SQLite (for user data and conversation history)
- **Vector Database**: ChromaDB (for optimized context retrieval)
- **AI Model**: OpenAI GPT-based API
- **Authentication**: OAuth2 with JWT-based authentication

### Key Functionalities

#### AI Conversation Handling

- Conversations are stored in SQLite and retrieved when needed.
- AI messages are embedded and stored in a vector database (ChromaDB) for efficient retrieval.
- The AI character remains in character, utilizing past messages for contextual responses.

#### User Authentication

- Users sign up and log in using JWT-based authentication.
- User session management ensures that only authorized users access their conversation history.

#### Character Management

- Users can retrieve and create AI characters.
- Dynamic character generation using OpenAI ensures that characters are immersive and unique, custom fit to the user's requirements.

### Deployment Details

- **Hosting**: Render.com
- **CI/CD**: Code is automatically deployed upon push to the production branch.

## Challenges and Solutions

### 1. Context Retrieval Optimization

**Challenge**: AI responses would cost more, due to sending back the entire conversation history.
**Solution**: Implemented a vector database (ChromaDB) to store embeddings of past messages and retrieve only the most relevant ones, for proper conversation context.

### 2. AI Response Accuracy

**Challenge**: The AI would sometimes break character or hallucinate responses.
**Solution**: Adjusted the prompt engineering to reinforce character consistency and fallback mechanisms.

### 3. Deployment Issues

**Challenge**: Initial deployment failed due to environment misconfigurations.
**Solution**: Standardized environment variables and dependencies.

## Future Improvements

- Custom embeddings instead of OpenAI embeddings to reduce API costs.
- Audio & video-based interactions for enhanced engagement.
- More sophisticated memory retention across sessions.
