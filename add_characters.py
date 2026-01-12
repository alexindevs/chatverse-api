import sys
import json
import asyncio
from pathlib import Path

sys.path.append(".")

from app.config.db import async_session, async_engine, Base
from app.models.user import User
from app.models.character import Character
from app.models.conversation import Conversation
from app.models.message import Message

async def insert_characters():
    """Create tables (if needed) and insert characters from `characters.json` into the database asynchronously."""
    data_file = Path("characters.json")
    if not data_file.is_file():
        print("Error: characters.json not found in project root.")
        sys.exit(1)

    try:
        with data_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            characters_data = data.get("characters", [])
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as db:
        try:
            for char_data in characters_data:
                new_char = Character(
                    name=char_data["name"],
                    nationality=char_data.get("nationality"),
                    profession=char_data.get("profession"),
                    description=char_data.get("description"),
                    image_url=char_data.get("image_url"),
                    background=char_data.get("background"),
                    personality_traits=";".join(char_data.get("personality_traits", [])),
                    quirks_habits=";".join(char_data.get("quirks_habits", [])),
                    example_sentences=";".join(char_data.get("example_sentences", [])),
                    motivations=char_data.get("motivations"),
                    is_personal_character=False,
                    owner_id=None
                )
                db.add(new_char)

            await db.commit()
            print("Characters inserted successfully!")
        except Exception as e:
            await db.rollback()
            print(f"Error inserting characters: {e}")
        finally:
            await db.close()

if __name__ == "__main__":
    asyncio.run(insert_characters())
