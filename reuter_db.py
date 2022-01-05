from enum import Enum
from pathlib import Path
import os
import re
from typing import Any, Dict, List, Tuple, Set

from database_builder import DatabaseBuilder
    
def archive_parser(file_path: str) -> Tuple[List[str], List[dict]]:
    text_lines = []
    texts = []
    with open(file_path, 'r') as fd:
        doctype = fd.readline()
        for line in fd:
            if line.startswith("<REUTERS"):
                if text_lines:
                    texts.append(" ".join(text_lines))
                    text_lines = []
                continue
            if line.startswith("</REUTERS"):
                continue
            text_lines += line
                    
    all_data = [parser(text) for text in texts]
    docs = [item[0] for item in all_data]
    metadata = [item[1] for item in all_data]
    
    return docs, metadata

    
def parser(text: str) -> Tuple[str, dict]:
    tags = extract_tag_names(text)
    tag_texts = {tag.lower():extract_tag_text(tag) for tag in tags}
    
    doc = tag_texts['body']
    tag_texts.pop('body')
    
    return doc, tag_texts

def extract_tag_names(text: str) -> Set[str]:
    patt = re.compile('<[A-Z]{2, 20}>')
    names = patt.findall(text)
    
    return set([n[1:-1] for n in names])

def extract_tag_text(tag: str) -> str:
    match = re.search(f'<{tag}>(.*)</{tag}>')
    if match is None:
        raise ValueError('Tag not found.')
    
    text = match.groups()[0]
    
    return text

    
def build_db()
    """
    Builds the cran database.
    """
    metadata: List[Dict[str, Any]] = []
    texts: List[str] = []
    
    reuter_files = []
    for root, _, files in os.walk("./test_collections/reuter/"):
        for file in files:
            reuter_files.append(os.path.join(root, file))
    
    for file_path in reuter_files:
        current_texts, current_metadatas = archive_parser(file_path)
        metadata.extend(current_metadatas)
        texts.extend(current_texts)
    
    DatabaseBuilder.build("reu", metadata, texts)