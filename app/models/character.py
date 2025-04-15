from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey
from ..config.db import Base
from sqlalchemy.orm import relationship

class Character(Base):
    __tablename__ = "characters"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    nationality = Column(String, nullable=True)
    profession = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    image_url = Column(String, nullable=True)

    background = Column(Text, nullable=True)
    personality_traits = Column(Text, nullable=True)
    motivations = Column(Text, nullable=True)
    quirks_habits = Column(Text, nullable=True)
    example_sentences = Column(Text, nullable=True)

    is_personal_character = Column(Boolean, default=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    conversations = relationship("Conversation", back_populates="character")