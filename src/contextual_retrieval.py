"""
Contextual Retrieval for RAG Pipeline

Based on Anthropic's research (September 2024):
- Generates LLM-based context for each chunk
- 67% reduction in top-20 chunk retrieval failures
- 35% reduction in top-5 failures
- Works with BM25 (53% reduction) and dense retrieval

Supports:
- Anthropic Claude (Haiku, Sonnet)
- OpenAI (GPT-4o-mini, GPT-4o)
- Local Legal LLMs (Saul-7B, Mistral-Legal-7B, Llama-3-8B)
  - Via Ollama (preferred - fast, simple)
  - Via Transformers (fallback - more complex)

Reference: https://www.anthropic.com/news/contextual-retrieval
"""

import logging
import os
import re
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time

# Import config
try:
    from .config import ContextGenerationConfig, resolve_model_alias
    from .cost_tracker import get_global_tracker
except ImportError:
    from config import ContextGenerationConfig, resolve_model_alias
    from cost_tracker import get_global_tracker

logger = logging.getLogger(__name__)


@dataclass
class ChunkContext:
    """Generated context for a chunk."""
    chunk_text: str
    context: str
    success: bool
    error: Optional[str] = None


class ContextualRetrieval:
    """
    Generate LLM-based context for document chunks.

    Replaces generic summaries (H-SAC) with chunk-specific context
    that explains what the chunk discusses within the broader document.

    Based on Anthropic research:
    - Context length: 50-100 words
    - Temperature: 0.3 (low for consistency)
    - Fast models preferred (Haiku, GPT-4o-mini)
    """

    def __init__(
        self,
        config: Optional[ContextGenerationConfig] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize contextual retrieval system.

        Args:
            config: ContextGenerationConfig (uses defaults if None)
            api_key: API key for cloud providers (optional, will use config if not provided)
        """
        self.config = config or ContextGenerationConfig()
        self.model = resolve_model_alias(self.config.model)

        # Initialize cost tracker
        self.tracker = get_global_tracker()

        # Extract Batch API config
        self.use_batch_api = self.config.use_batch_api
        self.batch_api_poll_interval = self.config.batch_api_poll_interval
        self.batch_api_timeout = self.config.batch_api_timeout

        # Use API key from config if not provided explicitly
        if api_key is None:
            api_key = self.config.api_key

        # Initialize LLM based on provider
        if self.config.provider == "anthropic":
            self._init_anthropic(api_key)
        elif self.config.provider == "openai":
            self._init_openai(api_key)
        elif self.config.provider == "local":
            self._init_local()
        else:
            raise ValueError(
                f"Unsupported provider: {self.config.provider}. "
                f"Supported: 'anthropic', 'openai', 'local'"
            )

        logger.info(
            f"ContextualRetrieval initialized: "
            f"provider={self.config.provider}, model={self.model}"
        )

    def _init_anthropic(self, api_key: Optional[str]):
        """Initialize Anthropic Claude client."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for Claude. "
                "Install with: pip install anthropic"
            )

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. "
                "Set ANTHROPIC_API_KEY env var or pass api_key parameter."
            )

        self.client = Anthropic(api_key=api_key)
        self.provider_type = "anthropic"
        logger.info("Anthropic client initialized")

    def _init_openai(self, api_key: Optional[str]):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI. "
                "Install with: pip install openai"
            )

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. "
                "Set OPENAI_API_KEY env var or pass api_key parameter."
            )

        self.client = OpenAI(api_key=api_key)
        self.provider_type = "openai"
        logger.info("OpenAI client initialized")

    def _init_local(self):
        """
        Initialize local LLM.

        Try Ollama first (preferred), fallback to Transformers.
        """
        # Try Ollama first (fast, simple)
        if self._try_ollama():
            return

        # Fallback to Transformers (slower but always works)
        self._init_transformers()

    def _try_ollama(self) -> bool:
        """Try to initialize Ollama client."""
        try:
            import requests

            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=2)

            if response.status_code == 200:
                # Ollama is running, check if model is available
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]

                # Map our model name to Ollama model name
                ollama_model = self._get_ollama_model_name()

                if ollama_model in model_names:
                    self.provider_type = "ollama"
                    self.ollama_model = ollama_model
                    logger.info(f"Using Ollama with model: {ollama_model}")
                    return True
                else:
                    logger.warning(
                        f"Ollama running but model '{ollama_model}' not found. "
                        f"Available: {model_names}. "
                        f"Run: ollama pull {ollama_model}"
                    )
                    return False
            else:
                logger.warning("Ollama API returned non-200 status")
                return False

        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False

    def _get_ollama_model_name(self) -> str:
        """Map our model name to Ollama model name."""
        ollama_mapping = {
            "Equall/Saul-7B-Instruct-v1": "saul",
            "meta-llama/Meta-Llama-3-8B-Instruct": "llama3:8b",
            "mistralai/Mistral-7B-Instruct-v0.3": "mistral:7b",
        }
        return ollama_mapping.get(self.model, self.model)

    def _init_transformers(self):
        """Initialize Transformers-based local LLM."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch required for local models. "
                "Install with: pip install transformers torch"
            )

        logger.info(f"Loading local model: {self.model}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.local_model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=torch.float16,
            device_map="auto",  # Auto-select GPU/CPU
            low_cpu_mem_usage=True
        )

        self.provider_type = "transformers"
        logger.info(f"Local model loaded: {self.model}")

    def generate_context(
        self,
        chunk: str,
        document_summary: Optional[str] = None,
        section_title: Optional[str] = None,
        section_path: Optional[str] = None,
        preceding_chunk: Optional[str] = None,
        following_chunk: Optional[str] = None
    ) -> str:
        """
        Generate context for a single chunk.

        Args:
            chunk: The text chunk to contextualize
            document_summary: Summary of the whole document
            section_title: Title of the section this chunk belongs to
            section_path: Hierarchical path (e.g., "Ch1 > Sec1.1 > Subsec1.1.1")
            preceding_chunk: The chunk that comes before (if available)
            following_chunk: The chunk that comes after (if available)

        Returns:
            Generated context (50-100 words)
        """
        prompt = self._build_context_prompt(
            chunk=chunk,
            document_summary=document_summary,
            section_title=section_title,
            section_path=section_path,
            preceding_chunk=preceding_chunk,
            following_chunk=following_chunk
        )

        try:
            if self.provider_type == "anthropic":
                context = self._generate_with_anthropic(prompt)
            elif self.provider_type == "openai":
                context = self._generate_with_openai(prompt)
            elif self.provider_type == "ollama":
                context = self._generate_with_ollama(prompt)
            elif self.provider_type == "transformers":
                context = self._generate_with_transformers(prompt)
            else:
                raise ValueError(f"Unknown provider type: {self.provider_type}")

            return context.strip()

        except Exception as e:
            logger.error(f"Context generation failed: {e}")
            raise

    def _build_context_prompt(
        self,
        chunk: str,
        document_summary: Optional[str],
        section_title: Optional[str],
        section_path: Optional[str],
        preceding_chunk: Optional[str] = None,
        following_chunk: Optional[str] = None
    ) -> str:
        """
        Build prompt for context generation.

        Based on Anthropic's recommended format with surrounding chunks.
        """
        # Build context information
        context_info = []

        if document_summary:
            context_info.append(f"Document summary: {document_summary}")

        if section_path:
            context_info.append(f"Section hierarchy: {section_path}")
        elif section_title:
            context_info.append(f"Section: {section_title}")

        context_block = "\n".join(context_info) if context_info else "No additional context available."

        # Escape chunk content to prevent prompt injection
        chunk_escaped = self._escape_xml_tags(chunk)

        # Build surrounding chunks section
        surrounding_chunks_block = ""
        if self.config.include_surrounding_chunks and (preceding_chunk or following_chunk):
            surrounding_parts = []
            if preceding_chunk:
                preceding_escaped = self._escape_xml_tags(preceding_chunk[:500])
                surrounding_parts.append(f"<preceding_chunk>\n{preceding_escaped}\n</preceding_chunk>")
            if following_chunk:
                following_escaped = self._escape_xml_tags(following_chunk[:500])
                surrounding_parts.append(f"<following_chunk>\n{following_escaped}\n</following_chunk>")

            if surrounding_parts:
                surrounding_chunks_block = f"\n\nSurrounding chunks for additional context:\n" + "\n\n".join(surrounding_parts)

        # Build prompt (Anthropic format)
        prompt = f"""<document>
{context_block}{surrounding_chunks_block}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_escaped}
</chunk>

Please give a short succinct context (50-100 words) to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        return prompt

    def _generate_with_anthropic(self, prompt: str) -> str:
        """
        Generate context using Anthropic Claude.

        Includes retry logic with exponential backoff for rate limits.
        """
        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )

                # Track cost
                self.tracker.track_llm(
                    provider="anthropic",
                    model=self.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    operation="context"
                )

                return response.content[0].text
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a rate limit error
                if "rate" in error_str or "429" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                # Re-raise if not rate limit or last attempt
                logger.error(f"Anthropic API error: {self._sanitize_error(str(e))}")
                raise

    def _generate_with_openai(self, prompt: str) -> str:
        """
        Generate context using OpenAI.

        Includes retry logic with exponential backoff for rate limits.
        """
        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                # GPT-5 and O-series models use max_completion_tokens instead of max_tokens
                # GPT-5 models only support temperature=1.0 (default)
                if self.model.startswith(('gpt-5', 'o1', 'o3', 'o4')):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1.0,  # GPT-5 only supports default temperature
                        max_completion_tokens=self.config.max_tokens
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )

                # Track cost
                self.tracker.track_llm(
                    provider="openai",
                    model=self.model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    operation="context"
                )

                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a rate limit error
                if "rate" in error_str or "429" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                # Re-raise if not rate limit or last attempt
                logger.error(f"OpenAI API error: {self._sanitize_error(str(e))}")
                raise

    def _generate_with_ollama(self, prompt: str) -> str:
        """Generate context using Ollama."""
        import requests

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            },
            timeout=30
        )

        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise RuntimeError(f"Ollama API error: {response.status_code}")

    def _generate_with_transformers(self, prompt: str) -> str:
        """Generate context using Transformers."""
        import torch

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.local_model.device)

        # Generate
        with torch.no_grad():
            outputs = self.local_model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (after prompt)
        context = generated[len(prompt):].strip()

        return context

    def _sanitize_error(self, error_msg: str) -> str:
        """
        Sanitize error messages to remove potential API keys.

        Args:
            error_msg: Raw error message

        Returns:
            Sanitized error message with API keys masked
        """
        # Mask Anthropic API keys (sk-ant-...)
        error_msg = re.sub(r'sk-ant-[a-zA-Z0-9_-]{32,}', 'sk-ant-***', error_msg)
        # Mask OpenAI API keys (sk-...)
        error_msg = re.sub(r'sk-[a-zA-Z0-9]{32,}', 'sk-***', error_msg)
        # Mask generic bearer tokens
        error_msg = re.sub(r'Bearer [a-zA-Z0-9_-]{20,}', 'Bearer ***', error_msg)
        return error_msg

    def _escape_xml_tags(self, text: str) -> str:
        """
        Escape XML tags in text to prevent prompt injection.

        Args:
            text: Raw text that may contain XML tags

        Returns:
            Text with XML tags escaped
        """
        # Only escape if text contains problematic tags
        if '</chunk>' in text or '</document>' in text:
            text = text.replace('<', '&lt;').replace('>', '&gt;')
        return text

    # ===== OpenAI Batch API Methods (50% cost savings) =====

    def _create_batch_requests(
        self,
        chunks: List[Tuple[str, dict]]  # [(chunk_text, metadata), ...]
    ) -> list[dict]:
        """Create batch API requests for OpenAI Batch API."""
        batch_requests = []

        for idx, (chunk_text, metadata) in enumerate(chunks):
            # Build context prompt
            doc_summary = metadata.get("document_summary", "")
            section_title = metadata.get("section_title", "")
            section_path = metadata.get("section_path", "")

            # Build prompt (similar to generate_context but simpler)
            chunk_preview = chunk_text[:1500]  # Limit chunk size for context

            prompt = f"""Document: {doc_summary}

Section: {section_path or section_title}

Chunk content:
{chunk_preview}

Provide a brief context (50-100 words) explaining what this chunk discusses within the document."""

            request = {
                "custom_id": f"chunk_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature
                }
            }
            batch_requests.append(request)

        return batch_requests

    def _submit_batch(self, requests: list[dict]) -> str:
        """Submit batch job to OpenAI Batch API."""
        import tempfile
        import json

        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for request in requests:
                f.write(json.dumps(request) + '\n')
            jsonl_path = f.name

        # Upload file to OpenAI
        with open(jsonl_path, 'rb') as f:
            file_response = self.client.files.create(file=f, purpose='batch')

        # Create batch job
        batch_response = self.client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        logger.info(f"Batch job created: {batch_response.id} (file: {file_response.id})")

        # Track batch ID for cost
        self.tracker.add_batch_job(
            provider="openai",
            model=self.model,
            operation="context_generation",
            batch_id=batch_response.id
        )

        # Clean up temp file
        import os
        os.unlink(jsonl_path)

        return batch_response.id

    def _poll_batch(self, batch_id: str) -> dict:
        """Poll batch job status until completion."""
        import time
        start_time = time.time()
        poll_count = 0

        while True:
            poll_count += 1
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed > self.batch_api_timeout:
                raise TimeoutError(
                    f"Batch job {batch_id} did not complete within {self.batch_api_timeout}s"
                )

            # Poll status
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status

            logger.info(f"Batch {batch_id} status: {status} (poll {poll_count}, {elapsed:.0f}s elapsed)")

            if status == "completed":
                logger.info(f"✓ Batch job {batch_id} completed successfully")
                return batch
            elif status in ["failed", "expired", "cancelled"]:
                error_msg = f"Batch job {batch_id} {status}"
                if hasattr(batch, 'errors') and batch.errors:
                    error_msg += f": {batch.errors}"
                raise Exception(error_msg)
            elif status in ["validating", "in_progress", "finalizing"]:
                # Expected states - keep polling
                time.sleep(self.batch_api_poll_interval)
            else:
                # Unknown status - keep polling but warn
                logger.warning(f"Unknown batch status: {status}")
                time.sleep(self.batch_api_poll_interval)

    def _parse_batch_results(self, batch: dict, chunks: List[Tuple[str, dict]]) -> dict:
        """Parse batch results from OpenAI Batch API."""
        import json
        import tempfile

        # Download output file
        output_file_id = batch.output_file_id
        content = self.client.files.content(output_file_id)

        # Save to temp file and parse
        results_map = {}
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(content.read())
            temp_path = f.name

        # Parse JSONL results
        with open(temp_path, 'r') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    custom_id = result['custom_id']
                    idx = int(custom_id.split('_')[1])  # Extract index from "chunk_123"

                    # Extract context from response
                    if result['response']['status_code'] == 200:
                        body = result['response']['body']
                        context = body['choices'][0]['message']['content'].strip()
                        results_map[idx] = context
                    else:
                        logger.warning(f"Chunk {idx} failed: {result['response']}")
                        results_map[idx] = ""  # Empty context for failed chunks

        # Clean up
        import os
        os.unlink(temp_path)

        # Track cost for batch
        self.tracker.finalize_batch_job(batch.id, batch)

        return results_map

    def _generate_contexts_with_openai_batch(
        self,
        chunks: List[Tuple[str, dict]]
    ) -> dict:
        """Generate contexts using OpenAI Batch API (50% cost savings)."""
        logger.info(f"Using OpenAI Batch API: {len(chunks)} chunks (50% cost savings, async processing)")

        # Step 1: Create batch requests
        requests = self._create_batch_requests(chunks)

        # Step 2: Submit batch
        batch_id = self._submit_batch(requests)

        # Step 3: Poll for completion
        logger.info(
            f"Polling batch {batch_id} "
            f"(timeout: {self.batch_api_timeout}s, interval: {self.batch_api_poll_interval}s)"
        )
        batch = self._poll_batch(batch_id)

        # Step 4: Parse results
        results_map = self._parse_batch_results(batch, chunks)

        logger.info(f"✓ Batch API succeeded: {len(results_map)} contexts generated")
        return results_map

    def generate_contexts_batch(
        self,
        chunks: List[Tuple[str, dict]]
    ) -> List[ChunkContext]:
        """
        Generate contexts for multiple chunks in parallel.

        Tries OpenAI Batch API first (50% cheaper), falls back to parallel mode if:
        - Batch API disabled
        - Provider is not OpenAI
        - Batch job times out or fails

        Args:
            chunks: List of (chunk_text, metadata) tuples
                   metadata should contain: document_summary, section_title, etc.

        Returns:
            List of ChunkContext objects
        """
        # Try Batch API first (OpenAI only, 50% cheaper)
        contexts_map = {}
        if self.use_batch_api and self.config.provider == "openai":
            try:
                contexts_map = self._generate_contexts_with_openai_batch(chunks)
                logger.info(f"✓ Batch API succeeded: {len(contexts_map)} contexts generated")
            except Exception as e:
                # Classify error type for better user guidance
                error_str = str(e).lower()
                if "timeout" in error_str or "time" in error_str:
                    logger.warning(
                        f"⚠️  Batch API timeout ({e}). Batch processing took >12 hours. "
                        f"Falling back to fast mode (full price). Consider indexing smaller batches."
                    )
                    print("\n⚠️  ECO MODE TIMEOUT")
                    print(f"   Reason: Batch processing exceeded 12-hour limit")
                    print("   Fallback: Using fast mode (full API pricing)")
                    print("   Impact: 2x cost for this document")
                    print("   Suggestion: Index smaller documents or use fast mode\n")
                elif "auth" in error_str or "key" in error_str:
                    logger.error(
                        f"❌ Batch API authentication failed ({e}). Check OPENAI_API_KEY. "
                        f"Falling back to fast mode (full price)."
                    )
                    print("\n❌ ECO MODE UNAVAILABLE - Authentication Error")
                    print(f"   Reason: {e}")
                    print("   Fallback: Using fast mode (full API pricing)")
                    print("   Impact: 2x cost for this document")
                    print("   Fix: Verify OPENAI_API_KEY in .env\n")
                elif "rate" in error_str or "quota" in error_str:
                    logger.warning(
                        f"⚠️  Batch API rate limit/quota exceeded ({e}). "
                        f"Falling back to fast mode (full price). Try again later."
                    )
                    print("\n⚠️  ECO MODE UNAVAILABLE - Rate Limit")
                    print(f"   Reason: {e}")
                    print("   Fallback: Using fast mode (full API pricing)")
                    print("   Impact: 2x cost for this document")
                    print("   Suggestion: Wait before indexing more documents\n")
                else:
                    logger.warning(
                        f"⚠️  Batch API failed ({e}). "
                        f"Falling back to fast mode (full price). See error above for details."
                    )
                    print("\n⚠️  ECO MODE UNAVAILABLE")
                    print(f"   Reason: {e}")
                    print("   Fallback: Using fast mode (full API pricing)")
                    print("   Impact: 2x cost for this document\n")
                contexts_map = {}

        # If Batch API succeeded, build results from contexts_map
        if contexts_map:
            results = []
            for idx, (chunk_text, metadata) in enumerate(chunks):
                context = contexts_map.get(idx, "")
                results.append(ChunkContext(
                    chunk_text=chunk_text,
                    context=context,
                    success=bool(context),
                    error=None if context else "Batch API returned empty context"
                ))
            return results

        # Otherwise, use parallel mode (fallback or default)
        logger.info(f"Using parallel mode: {len(chunks)} chunks → {len(chunks)} API calls")
        results = []

        def generate_one(chunk_text: str, metadata: dict) -> ChunkContext:
            try:
                context = self.generate_context(
                    chunk=chunk_text,
                    document_summary=metadata.get("document_summary"),
                    section_title=metadata.get("section_title"),
                    section_path=metadata.get("section_path"),
                    preceding_chunk=metadata.get("preceding_chunk"),
                    following_chunk=metadata.get("following_chunk")
                )
                return ChunkContext(
                    chunk_text=chunk_text,
                    context=context,
                    success=True
                )
            except Exception as e:
                logger.error(f"Failed to generate context: {e}")
                return ChunkContext(
                    chunk_text=chunk_text,
                    context="",
                    success=False,
                    error=str(e)
                )

        # Generate in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Process in batches
            batch_size = self.config.batch_size

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                # Submit all tasks and preserve order
                futures = [
                    executor.submit(generate_one, chunk_text, metadata)
                    for chunk_text, metadata in batch
                ]

                # Wait for all futures in ORDER (not as_completed to preserve order)
                batch_results = [future.result() for future in futures]
                results.extend(batch_results)

                logger.info(f"Generated contexts for batch {i//batch_size + 1}")

        success_count = sum(1 for r in results if r.success)
        logger.info(
            f"Context generation complete: {success_count}/{len(chunks)} successful"
        )

        return results


# Example usage
if __name__ == "__main__":
    # Test with different providers
    configs = [
        ("Anthropic Haiku", ContextGenerationConfig(provider="anthropic", model="haiku")),
        ("OpenAI GPT-4o-mini", ContextGenerationConfig(provider="openai", model="gpt-4o-mini")),
        ("Local Saul-7B", ContextGenerationConfig(provider="local", model="saul-7b")),
    ]

    test_chunk = """
    The primary cooling circuit operates at a nominal pressure of 15.7 MPa.
    If pressure exceeds 16.2 MPa, the emergency core cooling system (ECCS)
    is automatically activated within 2 seconds.
    """

    test_metadata = {
        "document_summary": "VVER-1200 nuclear reactor safety specification",
        "section_title": "Pressure Limits and Emergency Response",
        "section_path": "Safety Parameters > Pressure Control > Primary Circuit"
    }

    for name, config in configs:
        print(f"\n=== Testing {name} ===")
        try:
            retrieval = ContextualRetrieval(config=config)
            context = retrieval.generate_context(
                chunk=test_chunk,
                **test_metadata
            )
            print(f"Context: {context}")
        except Exception as e:
            print(f"Error: {e}")
