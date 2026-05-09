#!/usr/bin/env python
"""
Azure OpenAI Integration Example with RAGAnything

This example shows how to use RAGAnything with Azure OpenAI endpoints
instead of the standard OpenAI API. Azure OpenAI is widely used in
enterprise and production environments.

Required environment variables (or .env file):
    AZURE_OPENAI_API_KEY        Your Azure OpenAI resource key
    AZURE_OPENAI_ENDPOINT       e.g. https://<resource>.openai.azure.com
    AZURE_OPENAI_API_VERSION    e.g. 2024-12-01-preview
    AZURE_CHAT_DEPLOYMENT       Deployment name for your chat model (e.g. gpt-4o)
    AZURE_VISION_DEPLOYMENT     Deployment name for your vision model (e.g. gpt-4o)
    AZURE_EMBED_DEPLOYMENT      Deployment name for your embedding model
    AZURE_EMBED_DIM             Embedding dimension (default: 3072 for text-embedding-3-large)

Quick start:
    cp env.example .env          # then fill in your Azure credentials
    python examples/azure_openai_example.py path/to/document.pdf
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path
from typing import Dict, List, Optional

import sys

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

load_dotenv(dotenv_path=".env", override=False)

# ── Azure configuration ────────────────────────────────────────────────────────
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")
AZURE_VISION_DEPLOYMENT = os.getenv("AZURE_VISION_DEPLOYMENT", "gpt-4o")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-large")
AZURE_EMBED_DIM = int(os.getenv("AZURE_EMBED_DIM", "3072"))


def _azure_client() -> AsyncAzureOpenAI:
    """Create a shared AsyncAzureOpenAI client."""
    return AsyncAzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )


def configure_logging():
    """Configure logging for the application."""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "azure_openai_example.log")
    )
    print(f"\nAzure OpenAI example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_file_path) or ".", exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                }
            },
        }
    )
    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_rag(
    file_path: str,
    output_dir: str,
    working_dir: str = "./rag_storage",
    parser: str = "mineru",
):
    """Process a document with RAGAnything using Azure OpenAI."""

    client = _azure_client()

    # ── LLM function ──────────────────────────────────────────────────────────
    async def llm_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Dict] = None,
        **kwargs,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=AZURE_CHAT_DEPLOYMENT,
            messages=messages,
            **{k: v for k, v in kwargs.items() if k not in ("messages",)},
        )
        return response.choices[0].message.content

    # ── Vision / VLM function ─────────────────────────────────────────────────
    async def vision_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Dict] = None,
        image_data: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        # Pre-built messages list (VLM-enhanced query path) — use directly
        if messages:
            response = await client.chat.completions.create(
                model=AZURE_VISION_DEPLOYMENT,
                messages=messages,
                **{k: v for k, v in kwargs.items() if k not in ("messages",)},
            )
            return response.choices[0].message.content

        # Single-image path (multimodal ingestion)
        if image_data:
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": content})

            response = await client.chat.completions.create(
                model=AZURE_VISION_DEPLOYMENT,
                messages=msgs,
                **{k: v for k, v in kwargs.items() if k not in ("messages",)},
            )
            return response.choices[0].message.content

        # Text-only fallback
        return await llm_model_func(
            prompt, system_prompt=system_prompt,
            history_messages=history_messages, **kwargs
        )

    # ── Embedding function ────────────────────────────────────────────────────
    async def _embed(texts: List[str]) -> List[List[float]]:
        response = await client.embeddings.create(
            model=AZURE_EMBED_DEPLOYMENT,
            input=texts,
        )
        return [item.embedding for item in response.data]

    embedding_func = EmbeddingFunc(
        embedding_dim=AZURE_EMBED_DIM,
        max_token_size=8192,
        func=_embed,
    )

    # ── RAGAnything setup ─────────────────────────────────────────────────────
    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=parser,
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    try:
        # Process the document
        await rag.process_document_complete(
            file_path=file_path,
            output_dir=output_dir,
            parse_method="auto",
        )

        # Example queries
        logger.info("\nQuerying processed document:")

        queries = [
            "What is the main content of the document?",
            "What are the key topics discussed?",
        ]
        for query in queries:
            logger.info(f"\n[Text Query]: {query}")
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"Answer: {result}")

    except Exception as e:
        logger.error(f"Error processing with RAG: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description="RAGAnything with Azure OpenAI"
    )
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--working-dir", "-w", default="./rag_storage",
        help="Working directory for RAG storage"
    )
    parser.add_argument(
        "--output", "-o", default="./output",
        help="Output directory for parsed files"
    )
    parser.add_argument(
        "--parser", default=os.getenv("PARSER", "mineru"),
        help="Parser to use: mineru, docling, or paddleocr"
    )
    args = parser.parse_args()

    if not AZURE_API_KEY or not AZURE_ENDPOINT:
        logger.error(
            "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set. "
            "Add them to your .env file or environment."
        )
        return

    os.makedirs(args.output, exist_ok=True)
    asyncio.run(
        process_with_rag(
            file_path=args.file_path,
            output_dir=args.output,
            working_dir=args.working_dir,
            parser=args.parser,
        )
    )


if __name__ == "__main__":
    configure_logging()
    print("RAGAnything — Azure OpenAI Example")
    print("=" * 40)
    main()
