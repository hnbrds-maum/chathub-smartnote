import uuid
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Optional

TARGET_ENTITY_LABELS = ["Person", "Organization", "Location", "Product", "Event"]

class EntityFormat(BaseModel):
    text: str = Field(description="Exact text of the entity")
    entity_type: Literal[*TARGET_ENTITY_LABELS] = Field(description="Entity label")

class SemanticFormat(BaseModel):
    topics: List[str] = Field(description="Describe topic in one word, from the given text")
    entities: List[EntityFormat] = Field(description="List of entities extracted from the given text.")

class VLMFormat(BaseModel):
    image_type: Literal["DECORATIVE", "INFORMATIVE"] = Field(description="DECORATIVE = purely stylistic image / INFORMATIVE = image which conveys docuement content")
    caption: str = Field(description="Concise 2-3 sentence caption in Korean. Omit if DECORATIVE")

class Chunk(BaseModel):
    # Basic Elastic schema
    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    chunk_type: str = Literal["TEXT", "PICTURE", "TABLE"]
    content: str
    page_start: int
    page_end: int
    document_id: str

    # for GraphDB
    headings: List[str]
    entities: List[Dict]
    previous_id : Optional[str] = None # only for text chunk
    file_path: Optional[str] = None # only for image