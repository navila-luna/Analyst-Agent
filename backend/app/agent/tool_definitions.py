from app.document_repository import MAX_CHUNKS_PER_READ

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Semantically search for focused factual or procedural evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "A focused search query."}
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

LIST_DOCUMENTS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_documents",
        "description": "List readable documents and chunk counts when file scope is unclear.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

READ_DOCUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "read_document",
        "description": (
            "Read ordered chunks from one document for summaries, comparisons, "
            "transformations, or other tasks requiring broad coverage. Continue "
            "from next_chunk when has_more is true and more coverage is needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doc_name": {"type": "string"},
                "start_chunk": {"type": "integer", "minimum": 0},
                "max_chunks": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_CHUNKS_PER_READ,
                },
            },
            "required": ["doc_name", "start_chunk", "max_chunks"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

TOOLS = [SEARCH_TOOL, LIST_DOCUMENTS_TOOL, READ_DOCUMENT_TOOL]