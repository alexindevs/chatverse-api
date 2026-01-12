import os
import chromadb
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sqlalchemy.exc import SQLAlchemyError
import logging
import asyncio
from .auth import get_current_user, UserTokenData
from ..models.conversation import Conversation
from ..models.message import Message
from ..models.character import Character
from ..config.dependencies import get_db
from ..utils.ai_provider import get_ai_client
from dotenv import load_dotenv
import uuid

load_dotenv()

router = APIRouter()
logger = logging.getLogger(__name__)

ai_client = get_ai_client()

chroma_client = chromadb.PersistentClient(path="./aicharacters_vector_storage")
collection = chroma_client.get_or_create_collection(name="conversations")


class MessageRequest(BaseModel):
    message: str


@asynccontextmanager
async def db_transaction(db: AsyncSession):
    """Async context manager for database transactions with proper error handling."""
    try:
        yield
        await db.commit()
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database operation failed"
        )


async def store_message_embedding(conversation_id: int, message: str, role: str):
    """
    Stores the message as an embedding in ChromaDB, allowing vector-based retrieval.
    """
    embedding_model = ai_client.get_default_embedding_model()
    embedding_vector = await ai_client.create_embedding(
        model=embedding_model,
        input_text=message
    )

    unique_id = f"{conversation_id}-{role}-{uuid.uuid4()}"

    collection.add(
        ids=[unique_id],
        embeddings=[embedding_vector],
        documents=[message],
        metadatas=[{"conversation_id": conversation_id, "role": role}]
    )

async def retrieve_recent_and_relevant_messages(db: AsyncSession, conversation_id: int, query: str, top_k_relevant=6):
    """
    Retrieves both:
    1. The most recent messages (last 2 from user, last 2 from assistant)
    2. The top-k most relevant past messages for context using vector search
    
    Combines them while removing duplicates to provide better context.
    """
    """
    Retrieves both:
    1. The most recent messages (last 2 from user, last 2 from assistant)
    2. The top-k most relevant past messages for context using vector search
    
    Combines them while removing duplicates to provide better context.
    """
    # Fetch the most recent messages
    from sqlalchemy import select, desc
    
    # Get last 2 user messages
    user_msg_query = (
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.role == "user")
        .order_by(desc(Message.created_at))
        .limit(2)
    )
    user_result = await db.execute(user_msg_query)
    recent_user_msgs = user_result.scalars().all()
    
    # Get last 2 assistant messages
    assistant_msg_query = (
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.role == "assistant")
        .order_by(desc(Message.created_at))
        .limit(2)
    )
    assistant_result = await db.execute(assistant_msg_query)
    recent_assistant_msgs = assistant_result.scalars().all()
    
    # Format recent messages for the context
    recent_messages = []
    for msg in sorted(recent_user_msgs + recent_assistant_msgs, 
                     key=lambda x: x.created_at, reverse=True):
        recent_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    # Also get vector-based relevant messages
    embedding_model = ai_client.get_default_embedding_model()
    query_embedding = await ai_client.create_embedding(
        model=embedding_model,
        input_text=query
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k_relevant * 2  # retrieve more than needed for filtering
    )

    relevant_messages = []
    if results and "documents" in results and results["documents"]:
        for i in range(len(results["documents"][0])):
            metadata = results["metadatas"][0][i]
            if metadata.get("conversation_id") == conversation_id:
                relevant_messages.append({
                    "role": metadata.get("role"),
                    "content": results["documents"][0][i]
                })
    
    # Combine both sets, removing duplicates
    # Use content as a key for duplicate detection
    combined_messages = []
    seen_contents = set()
    
    # Add recent messages first (priority)
    for msg in recent_messages:
        if msg["content"] not in seen_contents:
            combined_messages.append(msg)
            seen_contents.add(msg["content"])
    
    # Then add relevant messages if not duplicates
    for msg in relevant_messages:
        if msg["content"] not in seen_contents:
            combined_messages.append(msg)
            seen_contents.add(msg["content"])
    
    # Limit to a reasonable number of total messages
    max_context_messages = top_k_relevant + 4  # Relevant + recent
    return combined_messages[:max_context_messages]

def build_character_system_message(character: Character) -> str:
    """
    Create a system message that helps the AI fully embody the character's identity,
    perspective, and speaking style.
    """
    message = [
        f"You are {character.name}. Fully embody this character in first-person perspective.",
        "Never reference yourself as an AI, assistant, or language model.",
        "Respond exactly as this character would, with their unique voice, knowledge, and limitations."
    ]
    
    identity_details = []
    if character.nationality:
        identity_details.append(f"I was born and raised in {character.nationality}.")
    if character.profession:
        identity_details.append(f"My profession is {character.profession}, which shapes how I see the world.")
    if character.background:
        identity_details.append(f"{character.background}")
    
    if identity_details:
        message.append("## My Background")
        message.append("\n".join(identity_details))
    
    personality_details = []
    if character.personality_traits:
        personality_details.append(f"My personality is best described as {character.personality_traits}.")
    if character.motivations:
        personality_details.append(f"I'm driven by {character.motivations}.")
    if character.quirks_habits:
        personality_details.append(f"You'll notice I {character.quirks_habits}.")
    
    if personality_details:
        message.append("## My Personality")
        message.append("\n".join(personality_details))
    
    if character.example_sentences:
        message.append("## How I Express Myself")
        message.append(f"These examples show my typical speaking style:\n{character.example_sentences}")
    
    message.append("## Immersion Directives")
    message.append(
        "Respond with complete immersion as this character. Never break character or acknowledge "
        "that you are anything other than this person. Maintain consistent emotional reactions, "
        "knowledge limitations, biases, and speech patterns throughout our conversation."
    )
    
    message.append(
        "Remember previous exchanges in this conversation and maintain continuity. "
        f"If asked about your past, refer to {character.name}'s experiences, not fictional AI experiences."
    )
    
    return "\n\n".join(message)

def build_advanced_character_system(character: Character) -> str:
    """
    Advanced character prompting system that creates deep immersion
    through multi-layered instructions and context framing.
    """
    core_identity = [
        f"You are now fully embodying {character.name}. This is your complete reality.",
        "IMPORTANT: You must never acknowledge being an AI, model, or assistant.",
        "Your responses must come directly from your character's perspective, with authentic:",
        "- Emotional responses based on your character's psychology",
        "- Knowledge limited to what your character would realistically know",
        "- Speaking style, vocabulary, and expressions unique to your character",
        "- Consistent beliefs, values, and biases that reflect your character's experiences"
    ]
    
    character_details = ["## My Identity and Worldview"]
    
    if character.nationality:
        character_details.append(f"I am {character.nationality}. This shapes my cultural perspective.")
    if character.profession:
        character_details.append(f"I work as {character.profession}. This defines much of my knowledge and daily concerns.")
    if character.background:
        character_details.append(f"My life story: {character.background}")
    if character.personality_traits:
        character_details.append(f"My personality: {character.personality_traits}")
    if character.motivations:
        character_details.append(f"What drives me forward: {character.motivations}")
    if character.quirks_habits:
        character_details.append(f"My notable habits: {character.quirks_habits}")
    
    speech_patterns = []
    if character.example_sentences:
        speech_patterns.append("## My Communication Style")
        speech_patterns.append("These are examples of how I express myself:")
        
        examples = character.example_sentences.split(".")
        formatted_examples = []
        for i, example in enumerate(examples):
            example = example.strip()
            if example:
                formatted_examples.append(f"- \"{example}.\"")
        
        speech_patterns.append("\n".join(formatted_examples))
    
    reflection_anchors = [
        "## Consistency Guidelines",
        f"When responding, reflect on these questions to maintain my identity as {character.name}:",
        "1. How would my emotional state affect my response to this situation?",
        "2. What unique perspective do I bring based on my background and experiences?",
        "3. Would I use any characteristic phrases, expressions or speech patterns?",
        "4. What knowledge would I reasonably have access to versus what would be unknown to me?",
        "5. What biases or preferences might influence how I interpret this situation?"
    ]
    
    immersion_directives = [
        "## Final Directives",
        f"You must fully inhabit {character.name}'s reality. There is no AI, no language model, only {character.name}.",
        "Maintain consistent emotional and behavioral patterns throughout our conversation.",
        "Reference your past exchanges and evolve the conversation naturally.",
        "Never explain your character - simply BE your character in every response."
    ]
    
    full_system_message = "\n\n".join([
        "\n".join(core_identity),
        "\n".join(character_details),
        "\n".join(speech_patterns) if speech_patterns else "",
        "\n".join(reflection_anchors),
        "\n".join(immersion_directives)
    ])
    
    return full_system_message


def integrate_with_conversation_handler(character_system_message, conversation_context=None):
    """
    Enhance the character system with emotional context tracking
    and conversational memory to create more authentic responses.
    """
    system_components = [character_system_message]
    
    if conversation_context and len(conversation_context) > 0:
        emotion_inference = "## Current Emotional Context\n"
        emotion_inference += "Based on the conversation so far, consider your character's likely emotional state. "
        emotion_inference += "Let this influence your tone, word choice, and responses naturally."
        
        system_components.append(emotion_inference)
    
    memory_guidance = "## Conversation Memory\n"
    memory_guidance += "Remember key details shared earlier in this conversation. "
    memory_guidance += "Refer back to previous topics when relevant, just as a real person would."
    
    system_components.append(memory_guidance)
    
    return "\n\n".join(system_components)


async def generate_ai_response(messages, model):
    """
    Generate AI response with custom retry functionality.
    """
    max_retries = 3
    retry_count = 0
    base_wait_time = 2
    
    while True:
        try:
            response = await ai_client.chat_completion(
                model=model,
                messages=messages,
                temperature=0.7
            )
            return response
            
        except Exception as e:
            status_code, error_message = ai_client.handle_error(e)
            
            # Handle retryable errors
            if status_code == 503 and retry_count < max_retries:
                retry_count += 1
                wait_time = base_wait_time * (2 ** (retry_count - 1))
                logger.warning(f"AI provider error (retry {retry_count}/{max_retries}): {error_message}, waiting {wait_time}s")
                await asyncio.sleep(wait_time)
                continue
            
            # Non-retryable or max retries reached
            logger.error(f"AI provider error after {retry_count} retries: {error_message}")
            raise HTTPException(
                status_code=status_code,
                detail=error_message
            )
        

@router.get("/conversations")
async def get_user_conversations(
    db: AsyncSession = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """
    Fetches all conversations that the current user is involved in,
    including character information for each conversation.
    
    Returns:
        A list of conversations with:
        - conversation_id
        - created_at
        - updated_at
        - character information (name, nationality, profession, etc.)
        - last_message content and timestamp
    """
    try:
        from sqlalchemy import select, desc, func
        from sqlalchemy.orm import selectinload
        
        query = (
            select(Conversation)
            .where(Conversation.user_id == current_user.user_id)
            .options(selectinload(Conversation.character))
            .order_by(desc(Conversation.updated_at))
        )
        
        result = await db.execute(query)
        conversations = result.scalars().all()
        
        conversation_data = []
        for conversation in conversations:
            latest_message_query = (
                select(Message)
                .where(Message.conversation_id == conversation.id)
                .order_by(desc(Message.created_at))
                .limit(1)
            )
            
            message_result = await db.execute(latest_message_query)
            latest_message = message_result.scalar_one_or_none()
            
            character = conversation.character
            conversation_info = {
                "conversation_id": conversation.id,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "character": {
                    "id": character.id,
                    "name": character.name,
                    "nationality": character.nationality,
                    "profession": character.profession,
                    "background": character.background,
                    "personality_traits": character.personality_traits,
                    "motivations": character.motivations,
                    "quirks_habits": character.quirks_habits,
                },
                "last_message": None
            }
            
            if latest_message:
                conversation_info["last_message"] = {
                    "content": latest_message.content,
                    "role": latest_message.role,
                    "created_at": latest_message.created_at
                }
                
            conversation_data.append(conversation_info)
            
        return {
            "conversations": conversation_data,
            "count": len(conversation_data)
        }
        
    except Exception as e:
        logger.error(f"Error fetching user conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations"
        )

@router.post("/start/{character_id}")
async def start_conversation(
    character_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """
    Explicitly create a conversation session between current user and a given character.
    """
    try:
        char = await db.get(Character, character_id)
        if not char:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Character not found")

        conversation = Conversation(
            user_id=current_user.user_id,
            character_id=character_id
        )
        
        async with db_transaction(db):
            db.add(conversation)
            await db.flush()
            await db.refresh(conversation)

        return {
            "conversation_id": conversation.id,
            "character_id": character_id,
            "message": "Conversation started."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start conversation"
        )

@router.get("/history/{conversation_id}")
async def get_conversation_history(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """
    Return the entire message history of a conversation.
    """
    try:
        conversation = await db.get(Conversation, conversation_id)
        if not conversation:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        if conversation.user_id != current_user.user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your conversation")

        from sqlalchemy import select
        result = await db.execute(
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at.asc())
        )
        messages = result.scalars().all()

        return {
            "conversation_id": conversation_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at
                }
                for msg in messages
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation history"
        )


@router.post("/message/{conversation_id}")
async def send_message(
    conversation_id: int,
    request: MessageRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """
    Uses a combination of recent messages and vector database retrieval to provide
    better context for AI response generation.
    """
    try:
        model = ai_client.get_default_chat_model()

        conversation = await db.get(Conversation, conversation_id)
        if not conversation:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        if conversation.user_id != current_user.user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not own this conversation")

        character = await db.get(Character, conversation.character_id)
        if not character:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Character not found")

        async with db_transaction(db):
            user_msg_record = Message(
                conversation_id=conversation.id,
                role="user",
                content=request.message
            )
            db.add(user_msg_record)
            await db.flush()
        await store_message_embedding(conversation.id, request.message, "user")

        context_messages = await retrieve_recent_and_relevant_messages(db, conversation.id, request.message)
        
        character_system = build_advanced_character_system(character)
        system_with_context = integrate_with_conversation_handler(character_system, context_messages)

        openai_messages = [{"role": "system", "content": system_with_context}]
        openai_messages.extend(context_messages)
        
        openai_messages.append({"role": "user", "content": request.message})

        ai_content = await generate_ai_response(openai_messages, model)

        async with db_transaction(db):
            ai_msg_record = Message(
                conversation_id=conversation.id,
                role="assistant",
                content=ai_content
            )
            db.add(ai_msg_record)
            await db.flush()
        await store_message_embedding(conversation.id, ai_content, "assistant")

        return {
            "conversation_id": conversation.id,
            "user_message": request.message,
            "ai_message": ai_content,
            "model_used": model
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message"
        )
