import os
import io
import uuid
import json
import logging
import yaml
import base64
from pathlib import Path
from PIL import Image

import requests
from dotenv import load_dotenv
import pandas as pd
import datetime

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Optional

from docling_core.types.doc import PictureItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    RapidOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database.schema import *

#IMG_DIR = Path("./imgs")
#IMG_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 256
CHUNK_OVERLAP = 32

MARKDOWN_HEADERS_TO_SPLIT = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
    ('####', "H4"),
]

class MarkdownSection(BaseModel):
    id: str
    sequence: int
    header: str
    content: str

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class OpenaiTagger:
    def extract_semantics(self, content):
        _response = client.responses.parse(
            model="gpt-4o",
            input=[
                {"role" : "system", "content" : "Extract the semantics from the given text. Follow the language of given text."},
                {"role" : "user", "content" : content},
            ],
            text_format=SemanticFormat,
        ).output_parsed
        return _response.topics, [x.model_dump() for x in _response.entities]
    
    def tag_caption(self, b64_image):
        response = client.responses.parse(
            model="gpt-4.1",
            input=[
                { "role": "system", "content" : "Classify whether given image is a DECORATIVE(=purely stylistic image), or INFORMATIVE(=conveys document content." },
                {
                    "role": "user",
                     "content" : [
                         { "type" : "input_text", "text" : "Describe the image in three sentences. Be consise and accurate" },
                         { "type" : "input_image",
                          "image_url" : b64_image }
                     ]
                }
            ],
            text_format=VLMFormat,
        ).output_parsed
        return response.image_type, response.caption
    

class DocumentParser:
    DEFAULT_IMAGE_RESOLUTION_SCALE = 1.0
    
    def __init__(self, document_path,
                 layout_model="ds4sd/docling-models",
                 embed_model="sentence-transformers/all-MiniLM-L6-v2",
                 document_id=None):
        self.document_path = document_path
        self.layout_model = layout_model
        self.embed_model = embed_model
        self.document_id = document_id or uuid.uuid4().hex
        self.document = self._convert_document()

    def _convert_document(self):
        ocr_opts = EasyOcrOptions(lang=["ko", "en"])
        pdf_pipeline_opts = PdfPipelineOptions(
            layout_model=self.layout_model,
            do_ocr=True,
            ocr_options=ocr_opts,
            include_images=True
        )
        pdf_pipeline_opts.images_scale = self.DEFAULT_IMAGE_RESOLUTION_SCALE
        pdf_pipeline_opts.generate_page_images = True
        pdf_pipeline_opts.do_table_structure = True
        pdf_pipeline_opts.table_structure_options.do_cell_matching = True
        pdf_pipeline_opts.accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.AUTO
        )
        
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_opts)
            }
        )
        return doc_converter.convert(self.document_path).document


    def get_text_chunks(self, md_sections, tag_semantics=False):
        prev_chunk_id = None
        text_chunks = []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        for section in md_sections:
            splits = text_splitter.split_text(section.content)
            for split in splits:
                chunk_id = uuid.uuid4().hex
                metadata = {
                    "document_id" : self.document_id,
                    "heading_id" : section.id,
                    "chunk_id" : chunk_id,
                    "chunk_type" : "TEXT",
                    "previous_id" : prev_chunk_id,
                }
                if tag_semantics:
                    topics, entities = self.tagger.extract_semantics(split)
                    metadata['topics'] = topics
                    metadata['entities'] = entities
                text_chunks.append(Document(
                    page_content=split,
                    metadata=metadata
                ))
                prev_chunk_id = chunk_id
        return text_chunks
        

    def get_image_chunks(self, tag_semantics=False):
        image_chunks = []
        
        for item, _ in self.document.iterate_items(with_groups=False):
            if getattr(item, "label", None) == DocItemLabel.SECTION_HEADER:
                cur_headings = [item.text.strip()]
        
            elif getattr(item, "label", None) in [DocItemLabel.PICTURE, DocItemLabel.TABLE]:
                pil_img = item.get_image(self.document)
                image_type, caption = self.tagger.tag_caption(pil_to_base64(pil_img))
                if image_type == "DECORATIVE":
                    continue

                chunk_id = uuid.uuid4().hex
                #img_path = IMG_DIR / f"{chunk_id}.png"
                #pil_img.save(img_path, "PNG") # save image to {img_path}
                metadata = {
                    "document_id" : self.document_id,
                    "heading_id" : None,
                    "chunk_id": chunk_id,
                    "chunk_type": item.label.upper(),
                    #"file_path" : img_path.as_posix()
                }

                if tag_semantics: 
                    topics, entities = self.tagger.extract_semantics(caption)
                    metadata['topics'] = topics
                    metadata['entities'] = entities
                
                image_chunks.append(Document(
                    page_content=caption,
                    metadata=metadata
                ))
        return image_chunks
        
    
    def get_chunk(self, tag_semantics=False, parse_image=False, markdown=None):
        if not markdown:
            markdown = self.get_markdown()

        tag_semantics = False # DO NOT USE SEMNATIC TAGGING IN SMARTNOTE SERVICE
        chunks = self.get_text_chunks(markdown, tag_semantics)
        if parse_image:
            chunks.extend(self.get_image_chunks(tag_semantics))
        return chunks
    
    
    def get_markdown(self):
        markdown =  self.document.export_to_markdown()
        markdown_splitter = MarkdownHeaderTextSplitter(MARKDOWN_HEADERS_TO_SPLIT)
        md_header_splits = markdown_splitter.split_text(markdown)
        
        result = []
        for idx, section in enumerate(md_header_splits):
            result.append(
                MarkdownSection(
                    id=uuid.uuid4().hex,
                    sequence=idx,
                    header=find_header_from_metadata(
                        section.metadata, Path(self.document_path).stem),
                    content=section.page_content
                )
            )
        return result

def find_header_from_metadata(metadata, default="# "):
    if 'H4' in metadata:
        return f"#### {metadata['H4']}"
    elif 'H3' in metadata:
        return f"### {metadata['H3']}"
    elif 'H2' in metadata:
        return f"## {metadata['H2']}"
    elif 'H1' in metadata:
        return f"# {metadata['H1']}"
    return f'# {default}'

    
def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64_str}"