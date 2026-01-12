import os
import json
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import ValidationError
from typing import List, Any, Dict
from sqlalchemy import or_
from sqlalchemy.future import select
from .auth import get_current_user, UserTokenData
from ..config.dependencies import get_db
from ..models.character import Character
from ..utils.ai_provider import get_ai_client
from ..schemas.character import (
    CharacterCreate,
    CharacterUpdate,
    CharacterOut
)

router = APIRouter()

ai_client = get_ai_client()


# ----------------------------
# Helper Functions
# ----------------------------
def serialize_list_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Python list fields (personality_traits, quirks_habits, example_sentences)
    into a semicolon-separated string for storage in the DB.
    """
    list_fields = ["personality_traits", "quirks_habits", "example_sentences"]
    for field in list_fields:
        value = data.get(field)
        if isinstance(value, list):
            # Join list elements with a semicolon.
            data[field] = ";".join(value)
    return data


def prepare_character_response(char: Character) -> CharacterOut:
    """
    Build a response model (CharacterOut) from the ORM object without modifying it.
    Converts semicolon-separated string fields into lists.
    """
    response_data = {
        "id": char.id,
        "name": char.name,
        "nationality": char.nationality,
        "profession": char.profession,
        "description": char.description,
        "image_url": char.image_url,
        "background": char.background,
        "personality_traits": char.personality_traits.split(";") if char.personality_traits else [],
        "motivations": char.motivations,
        "quirks_habits": char.quirks_habits.split(";") if char.quirks_habits else [],
        "example_sentences": char.example_sentences.split(";") if char.example_sentences else [],
        "is_personal_character": char.is_personal_character,
        "owner_id": char.owner_id,
    }
    return CharacterOut(**response_data)


@router.get("/", response_model=List[CharacterOut])
async def get_characters(
    db: AsyncSession = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """
    Fetch only:
    - Public (non-personal) characters (is_personal_character=False), or
    - Personal characters owned by the current user.
    """
    print(current_user, current_user.user_id)
    print('Entering get_characters endpoint')
    stmt = select(Character).where(
        or_(
            Character.is_personal_character == False,
            Character.owner_id == current_user.user_id
        )
    )

    result = await db.execute(stmt)
    chars = result.scalars().all()

    print('Fetched data; preparing response without modifying ORM objects')
    response_chars = [prepare_character_response(char) for char in chars]
    return response_chars


@router.get("/{character_id}", response_model=CharacterOut)
async def get_character(
    character_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    char = await db.get(Character, character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")
    return prepare_character_response(char)


@router.post("/", response_model=CharacterOut, status_code=status.HTTP_201_CREATED)
async def create_character(
    new_char: CharacterCreate,
    db: AsyncSession = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """
    Create a new character, personal to the user.
    """
    new_char.is_personal_character = True
    new_char.owner_id = current_user.user_id

    char_data = new_char.dict()
    char_data = serialize_list_fields(char_data)

    char = Character(**char_data)
    db.add(char)
    await db.commit()
    await db.refresh(char)

    return prepare_character_response(char)


@router.put("/{character_id}", response_model=CharacterOut)
async def update_character(
    character_id: int,
    char_update: CharacterUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """
    Update an existing character's fields, including potential list fields.
    """
    char = await db.get(Character, character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    if char.is_personal_character and char.owner_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to modify this character.")

    update_data = char_update.dict(exclude_unset=True)
    update_data = serialize_list_fields(update_data)

    for field, value in update_data.items():
        setattr(char, field, value)

    await db.commit()
    await db.refresh(char)
    return prepare_character_response(char)


@router.patch("/{character_id}/avatar", response_model=CharacterOut)
async def update_character_avatar(
    character_id: int,
    payload: dict,
    db: AsyncSession = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """
    Updates only the `image_url` for a personal character that the user owns.
    If the character is not personal or the user doesn't own it, raises 403.
    """
    char = await db.get(Character, character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    if not char.is_personal_character or char.owner_id != current_user.user_id:
        raise HTTPException(
            status_code=403,
            detail="You do not own this character or it's not personal."
        )

    new_image_url = payload.get("image_url")
    if not new_image_url:
        raise HTTPException(status_code=422, detail="No image_url provided.")

    char.image_url = new_image_url
    await db.commit()
    await db.refresh(char)
    return prepare_character_response(char)


@router.delete("/{character_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_character(
    character_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """
    Delete a character if the user owns it (if personal).
    """
    char = await db.get(Character, character_id)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    if char.is_personal_character and char.owner_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not allowed to delete this character.")

    await db.delete(char)
    await db.commit()
    return None


@router.post("/ai-generate", response_model=CharacterOut)
async def generate_character_automatically(
    preferences: dict,
    db: AsyncSession = Depends(get_db),
    current_user: UserTokenData = Depends(get_current_user)
):
    """
    Generate a character dynamically using AI, then save to the database.
    We'll ensure it's personal to this user, and handle list fields as semicolon-separated strings.
    """
    user_prompt = build_character_prompt(preferences)
    try:
        model = ai_client.get_default_chat_model()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert world-builder and character developer for an AI chat platform. "
                    "Your goal is to generate immersive, well-rounded, and believable fictional characters that feel real in an interactive storytelling experience. "
                    "Characters should have **depth, consistent personalities, and clear motivations** that align with their world—whether it's fantasy, sci-fi, historical, or modern-day settings."
                    "\n\n"
                    "Use the provided user preferences to construct a character that fits naturally into their environment. If some details are missing, "
                    "**fill in logical, compelling details** that enhance the character's backstory and role in their world. "
                    "Ensure that the character has quirks, flaws, and strengths that make interactions with them dynamic and unpredictable."
                    "\n\n"
                    "💡 **Your response MUST be a JSON object ONLY**, matching the following schema:"
                    "\n\n"
                    "Schema fields:\n"
                    "- `name` (str): The character's full name.\n"
                    "- `nationality` (str|None): The character's homeland, cultural background, or place of origin.\n"
                    "- `profession` (str|None): Their role in the world (e.g., doctor, hacker, detective, sorcerer, journalist, warrior, etc.).\n"
                    "- `description` (str|None): A vivid, engaging description of their appearance and personality.\n"
                    "- `image_url` (str|None): If applicable, provide an image link.\n"
                    "- `background` (str|None): A compelling backstory that explains who they are, their past experiences, and key life events that shaped them.\n"
                    "- `personality_traits` (list of strings|None): At least **5 defining personality traits** (e.g., analytical, loyal, arrogant, kind, impulsive).\n"
                    "- `motivations` (str|None): What drives them? Their ambitions, fears, or personal quests.\n"
                    "- `quirks_habits` (list of strings|None): Small behaviors that make them feel human (e.g., always checking their watch, biting their lip when thinking, talking to themselves).\n"
                    "- `example_sentences` (list of strings|None): At least **3 lines of dialogue** to reflect their voice and manner of speaking."
                    "\n\n"
                    "⚠️ **Important:**\n"
                    "- **DO NOT return null values** unless absolutely necessary.\n"
                    "- **Infer missing details logically** based on the character's traits.\n"
                    "- **Do NOT include markdown formatting, explanations, or extra text** outside the JSON object."
                )
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        response_content = await ai_client.chat_completion(
            model=model,
            messages=messages,
            temperature=0.7
        )
    except Exception as e:
        status_code, error_message = ai_client.handle_error(e)
        raise HTTPException(
            status_code=status_code,
            detail=error_message
        )

    raw_content = response_content.strip()

    try:
        new_char_data = CharacterCreate.parse_raw(raw_content)
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON from AI: {ve}"
        )

    new_char_data.is_personal_character = True
    new_char_data.owner_id = current_user.user_id

    data_dict = new_char_data.dict()
    data_dict = serialize_list_fields(data_dict)

    char_model = Character(**data_dict)
    db.add(char_model)
    await db.commit()
    await db.refresh(char_model)

    return prepare_character_response(char_model)


def build_character_prompt(preferences: dict) -> str:
    prompt_parts = [
        "You are creating a character for an AI-powered interactive chat experience. "
        "This character must feel real, with a well-developed personality, ambitions, and quirks that make them engaging to interact with. "
        "The character should fit seamlessly into their setting, whether it's fantasy, sci-fi, historical, or modern."
        "\n\n"
        "If any details are missing, use logical and creative reasoning to complete them in a way that enhances the character's depth. "
        "Ensure that the character has **flaws, motivations, and distinct traits** to avoid one-dimensional personalities."
        "\n\n"
        "📜 **Character Details:**"
    ]

    for k, v in preferences.items():
        if v:
            prompt_parts.append(f"- {k}: {v}")

    prompt_parts.append(
        "\n\n"
        "🎭 **Personality & Depth:**\n"
        "- Ensure the character has a **distinct voice**—someone users would want to keep talking to.\n"
        "- Their **motivations must be clear** (e.g., personal ambitions, fears, desires, or ongoing struggles).\n"
        "- They should have **flaws & quirks** that make them relatable and dynamic.\n"
        "- Their **dialogue should reflect their personality**, background, and emotional state.\n\n"
        "✍️ **Example Sentences:**\n"
        "- Provide at least three phrases they might say in conversation.\n"
        "- These should capture their **attitude, personality, and speaking style.**\n\n"
        "📜 **Final Output:**\n"
        "Return only a JSON object matching the expected schema. "
        "**Do not include any explanations, markdown, or extra text outside of the JSON.**\n\n"
        "Do not format the JSON object in markdown (don't do this - ```json ```). Return only the JSON object."
    )

    return "\n".join(prompt_parts)
