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

from schema import *

IMG_DIR = Path("./imgs")
IMG_DIR.mkdir(exist_ok=True)

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
                 embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.document_path = document_path
        self.layout_model = layout_model
        self.embed_model = embed_model
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
        
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_opts)
            }
        )
        return doc_converter.convert(self.document_path).document


    def get_text_chunks(self):
        prev_chunk_id = None
        text_chunks = []
        for i, chunk in enumerate(self.chunker.chunk(dl_doc=self.document)):
            _content = self.chunker.contextualize(chunk)
            _pages = {p.page_no for it in chunk.meta.doc_items for p in it.prov}
            topics, entities = self.tagger.extract_semantics(_content)
            chunk_id = uuid.uuid4().hex
            _chunk = {
                "document_id" : "TODO",
                "chunk_id": chunk_id,
                "chunk_type": "TEXT",
                "content": _content,
                "page_start": min(_pages),
                "page_end": max(_pages),
                "previous_id" : prev_chunk_id,
        
                "headings": chunk.meta.headings or [],
                "topics" : topics,
                "entities" : entities
            }
            text_chunks.append(Chunk(**_chunk))
            prev_chunk_id = chunk_id
        return text_chunks
        

    def get_image_chunks(self):
        image_chunks = []
        cur_headings = []
        
        for item, _ in self.document.iterate_items(with_groups=False):
            if getattr(item, "label", None) == DocItemLabel.SECTION_HEADER:
                cur_headings = [item.text.strip()]
        
            elif getattr(item, "label", None) in [DocItemLabel.PICTURE, DocItemLabel.TABLE]:
                pil_img = item.get_image(self.document)
                image_type, caption = self.tagger.tag_caption(pil_to_base64(pil_img))
                if image_type == "DECORATIVE":
                    continue

                chunk_id = uuid.uuid4().hex
                img_path = IMG_DIR / f"{chunk_id}.png"
                pil_img.save(img_path, "PNG") # save image to {img_path}
                topics, entities = self.tagger.extract_semantics(caption)
                _page = item.prov[0].page_no
                _chunk = {
                    "document_id" : "TODO",
                    "chunk_id": chunk_id,
                    "chunk_type": item.label.upper(),
                    "content": caption,
                    "page_start": _page,
                    "page_end": _page,
            
                    "headings" : cur_headings,
                    "topics" : topics,
                    "entities" : entities,
                    "file_path" : img_path.as_posix()
                }
                image_chunks.append(Chunk(**_chunk))
        return image_chunks
        
    
    def get_chunk(self, image=True):
        tokenizer: BaseTokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(self.embed_model)
        )
        self.chunker = HybridChunker(
            serializer_cfg=dict(
                include_headers=True,
                include_section_number=True,
                include_page=True,
            ),
            max_tokens=512,
            tokenizer=tokenizer
        )
        self.tagger = OpenaiTagger()

        chunks = self.get_text_chunks()
        if image:
            chunks.extend(self.get_image_chunks())
        return chunks
    
    def get_markdown(self):
        return self.document.export_to_markdown()


    
def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64_str}"