from typing import Optional, List
from pydantic import BaseModel, model_validator

class CharacterBase(BaseModel):
    name: str
    nationality: Optional[str] = None
    profession: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None

    background: Optional[str] = None
    personality_traits: Optional[List[str]] = None
    motivations: Optional[str] = None
    quirks_habits: Optional[List[str]] = None
    example_sentences: Optional[List[str]] = None

    is_personal_character: bool = False
    owner_id: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def check_owner_if_personal_character(cls, data):
        """Ensure `owner_id` is set if `is_personal_character` is True."""
        if isinstance(data, dict):  
            is_personal = data.get("is_personal_character")
            owner_id = data.get("owner_id")
        else:  
            is_personal = getattr(data, "is_personal_character", None)
            owner_id = getattr(data, "owner_id", None)
    
        if is_personal and owner_id is None:
            raise ValueError("owner_id must be provided if is_personal_character is True.")
    
        return data


class CharacterCreate(CharacterBase):
    """Schema for creating a new character."""
    pass


class CharacterUpdate(CharacterBase):
    """Schema for updating an existing character (partial updates)."""
    pass


class CharacterOut(CharacterBase):
    """Schema for reading/returning character data."""
    id: int

    class Config:
        from_attributes = True
