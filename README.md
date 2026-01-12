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
[AI Characters Platform](https://chatverse-eight.vercel.app)

API Documentation: [Swagger UI](https://chatverse-api-dqf6.onrender.com/docs)

## Technical Implementation

### Backend

- **Framework**: FastAPI
- **Database**: SQLite (for user data and conversation history)
- **Vector Database**: ChromaDB (for optimized context retrieval)
- **AI Model**: OpenAI GPT-based API or Google Gemini (configurable via environment variables)
- **Authentication**: OAuth2 with JWT-based authentication

### Environment Variables

The application supports both OpenAI and Google Gemini AI providers. Configure your `.env` file with the following variables:

**Required for AI Provider Selection:**
- `AI_PROVIDER`: Set to `"openai"` or `"gemini"` to switch between providers (default: `"openai"`)

**OpenAI Configuration (required if AI_PROVIDER=openai):**
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_DEFAULT_MODEL`: Chat model to use (default: `"gpt-4o"`)
- `OPENAI_EMBEDDING_MODEL`: Embedding model to use (default: `"text-embedding-ada-002"`)

**Gemini Configuration (required if AI_PROVIDER=gemini):**
- `GEMINI_API_KEY`: Your Google Gemini API key
- `GEMINI_DEFAULT_MODEL`: Chat model to use (default: `"gemini-pro"`)
- `GEMINI_EMBEDDING_MODEL`: Embedding model to use (default: `"models/embedding-001"`)

**Authentication:**
- `SECRET_KEY`: Secret key for JWT token signing
- `ALGORITHM`: JWT algorithm (default: `"HS256"`)
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time in minutes (default: `30`)

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
- Dynamic character generation using AI (OpenAI or Gemini) ensures that characters are immersive and unique, custom fit to the user's requirements.

### Deployment Details

- **Hosting**: Render.com
- **CI/CD**: Code is automatically deployed upon push to the production branch.

## Future Improvements

- Custom embeddings instead of OpenAI embeddings to reduce API costs.
- Audio & video-based interactions for enhanced engagement.
- More sophisticated memory retention across sessions.
