"""
Entity extraction from legal document chunks using LLM.

Extracts structured entities (Standards, Organizations, Dates, etc.) from text chunks
using few-shot prompting and parallel batch processing.
"""

import json
import logging
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import Entity, EntityType
from .config import EntityExtractionConfig


logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    LLM-based entity extractor for legal documents.

    Extracts entities from chunks using structured prompts and parallel processing.
    Supports entity normalization, confidence scoring, and provenance tracking.
    """

    def __init__(
        self,
        config: EntityExtractionConfig,
        api_key: Optional[str] = None,
    ):
        """
        Initialize entity extractor.

        Args:
            config: Entity extraction configuration
            api_key: API key for LLM provider (overrides config)
        """
        self.config = config
        self.api_key = api_key

        # Initialize LLM client based on provider
        self._initialize_llm_client()

        # Cache for extracted entities (chunk_id -> entities)
        self.cache: Dict[str, List[Entity]] = {}

    def _initialize_llm_client(self):
        """Initialize LLM client based on provider."""
        if self.config.llm_provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")

        elif self.config.llm_provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")

        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

    def extract_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[Entity]:
        """
        Extract entities from multiple chunks in parallel.

        Args:
            chunks: List of chunk dictionaries with 'id', 'content', 'metadata'

        Returns:
            List of extracted Entity objects
        """
        logger.info(f"Extracting entities from {len(chunks)} chunks...")

        # Process chunks in batches
        all_entities = []
        batches = [chunks[i:i + self.config.batch_size] for i in range(0, len(chunks), self.config.batch_size)]

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._extract_from_batch, batch): batch
                for batch in batches
            }

            for future in as_completed(futures):
                try:
                    batch_entities = future.result()
                    all_entities.extend(batch_entities)
                except Exception as e:
                    batch = futures[future]
                    logger.error(f"Failed to extract entities from batch: {e}")

        # Deduplicate entities across chunks
        if self.config.normalize_entities:
            all_entities = self._deduplicate_entities(all_entities)

        logger.info(f"Extracted {len(all_entities)} unique entities")
        return all_entities

    def _extract_from_batch(self, chunks: List[Dict[str, Any]]) -> List[Entity]:
        """Extract entities from a batch of chunks."""
        batch_entities = []

        for chunk in chunks:
            chunk_id = chunk.get("id", str(uuid.uuid4()))

            # Check cache
            if self.config.cache_results and chunk_id in self.cache:
                batch_entities.extend(self.cache[chunk_id])
                continue

            # Extract entities from chunk
            try:
                entities = self._extract_from_single_chunk(chunk)

                # Cache results
                if self.config.cache_results:
                    self.cache[chunk_id] = entities

                batch_entities.extend(entities)

            except Exception as e:
                logger.error(f"Failed to extract entities from chunk {chunk_id}: {e}")

        return batch_entities

    def _extract_from_single_chunk(self, chunk: Dict[str, Any]) -> List[Entity]:
        """Extract entities from a single chunk using LLM."""
        chunk_id = chunk.get("id", str(uuid.uuid4()))
        chunk_content = chunk.get("raw_content", chunk.get("content", ""))
        chunk_metadata = chunk.get("metadata", {})

        # Build extraction prompt
        prompt = self._build_extraction_prompt(chunk_content, chunk_metadata)

        # Call LLM
        response_text = self._call_llm(prompt)

        # Parse response
        entities = self._parse_llm_response(
            response_text,
            chunk_id=chunk_id,
            document_id=chunk_metadata.get("document_id"),
            section_path=chunk_metadata.get("section_path"),
        )

        return entities

    def _build_extraction_prompt(self, chunk_content: str, chunk_metadata: Dict[str, Any]) -> str:
        """Build entity extraction prompt for LLM."""
        # Enabled entity types
        entity_types_str = ", ".join([et.value for et in self.config.enabled_entity_types])

        prompt = f"""Extract structured entities from the following legal document text.

**Task**: Identify and extract all entities of the following types:
{entity_types_str}

**Entity Types**:
- standard: Standards like "GRI 306", "ISO 14001", "GRI 303: Water and Effluents 2018"
- organization: Organizations like "GSSB", "Global Reporting Initiative", "ISO"
- date: Dates like "2018-07-01", "1 July 2018", "July 2018"
- clause: Specific clauses like "Disclosure 306-3", "Section 8.2", "Requirement 303-1-a"
- topic: Topics like "waste management", "water", "effluents", "emissions"
- regulation: Regulations like "GDPR", "CCPA", "Basel Convention"
- contract: Contracts like "Master Service Agreement", "NDA"
- person: People like "Jane Smith", "Dr. John Doe"
- location: Locations like "European Union", "California", "United States"

**Document Text**:
{chunk_content}

**Metadata Context**:
- Section: {chunk_metadata.get('section_path', 'N/A')}
- Document ID: {chunk_metadata.get('document_id', 'N/A')}

**Output Format** (JSON array):
[
  {{
    "type": "standard",
    "value": "GRI 306: Effluents and Waste 2016",
    "normalized_value": "GRI 306",
    "confidence": 0.95,
    "context": "This Standard supersedes GRI 306: Effluents and Waste 2016."
  }},
  {{
    "type": "date",
    "value": "1 July 2018",
    "normalized_value": "2018-07-01",
    "confidence": 0.9,
    "context": "It includes revisions effective on 1 July 2018."
  }}
]

**Instructions**:
1. Extract ALL entities of the specified types
2. For each entity, provide:
   - type: Entity type (one of the types above)
   - value: Original text as it appears in the document
   - normalized_value: Standardized form (e.g., "GRI 306" for "GRI 306: Effluents and Waste 2016")
   - confidence: Extraction confidence (0.0-1.0)
   - context: Brief surrounding context (max 150 chars)
3. Normalize entity values:
   - Standards: Extract base identifier (e.g., "GRI 306")
   - Dates: Convert to ISO format (YYYY-MM-DD)
   - Organizations: Use official name
4. Only include entities with confidence >= {self.config.min_confidence}
5. Limit to {self.config.max_entities_per_chunk} entities
6. Return valid JSON array

"""

        # Add few-shot examples if enabled
        if self.config.include_examples:
            prompt += self._get_few_shot_examples()

        prompt += "\n**Output** (JSON array only, no other text):\n"

        return prompt

    def _get_few_shot_examples(self) -> str:
        """Get few-shot examples for entity extraction."""
        return """
**Example Input**:
"GRI 306: Waste 2020 supersedes GRI 306: Effluents and Waste 2016, which was issued by the Global Reporting Initiative (GRI) and became effective on 1 July 2018. This standard covers waste management topics including hazardous waste disposal."

**Example Output**:
[
  {"type": "standard", "value": "GRI 306: Waste 2020", "normalized_value": "GRI 306", "confidence": 0.95, "context": "GRI 306: Waste 2020 supersedes GRI 306"},
  {"type": "standard", "value": "GRI 306: Effluents and Waste 2016", "normalized_value": "GRI 306", "confidence": 0.95, "context": "supersedes GRI 306: Effluents and Waste 2016"},
  {"type": "organization", "value": "Global Reporting Initiative", "normalized_value": "GRI", "confidence": 0.9, "context": "issued by the Global Reporting Initiative (GRI)"},
  {"type": "organization", "value": "GRI", "normalized_value": "GRI", "confidence": 0.95, "context": "Global Reporting Initiative (GRI)"},
  {"type": "date", "value": "1 July 2018", "normalized_value": "2018-07-01", "confidence": 0.9, "context": "became effective on 1 July 2018"},
  {"type": "topic", "value": "waste management", "normalized_value": "waste management", "confidence": 0.85, "context": "covers waste management topics including"},
  {"type": "topic", "value": "hazardous waste disposal", "normalized_value": "hazardous waste disposal", "confidence": 0.8, "context": "topics including hazardous waste disposal"}
]

"""

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with retry logic."""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                if self.config.llm_provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.config.llm_model,
                        messages=[
                            {"role": "system", "content": "You are an expert at extracting structured entities from legal documents. Always return valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.config.temperature,
                        max_tokens=4000,
                    )
                    return response.choices[0].message.content.strip()

                elif self.config.llm_provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.config.llm_model,
                        max_tokens=4000,
                        temperature=self.config.temperature,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        system="You are an expert at extracting structured entities from legal documents. Always return valid JSON.",
                    )
                    return response.content[0].text.strip()

            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise

        raise Exception("Max retries exceeded for LLM call")

    def _parse_llm_response(
        self,
        response_text: str,
        chunk_id: str,
        document_id: Optional[str],
        section_path: Optional[str],
    ) -> List[Entity]:
        """Parse LLM response and create Entity objects."""
        entities = []

        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find JSON array directly
                json_text = response_text.strip()

            # Parse JSON
            entities_data = json.loads(json_text)

            if not isinstance(entities_data, list):
                logger.warning(f"Expected list of entities, got {type(entities_data)}")
                return []

            # Create Entity objects
            for entity_data in entities_data:
                try:
                    entity = self._create_entity_from_dict(
                        entity_data,
                        chunk_id=chunk_id,
                        document_id=document_id,
                        section_path=section_path,
                    )

                    # Filter by confidence
                    if entity.confidence >= self.config.min_confidence:
                        entities.append(entity)

                except Exception as e:
                    logger.warning(f"Failed to create entity from data {entity_data}: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")

        return entities

    def _create_entity_from_dict(
        self,
        entity_data: Dict[str, Any],
        chunk_id: str,
        document_id: Optional[str],
        section_path: Optional[str],
    ) -> Entity:
        """Create Entity object from dictionary."""
        # Parse entity type
        entity_type_str = entity_data.get("type", "").lower()
        try:
            entity_type = EntityType(entity_type_str)
        except ValueError:
            raise ValueError(f"Invalid entity type: {entity_type_str}")

        # Check if enabled
        if entity_type not in self.config.enabled_entity_types:
            raise ValueError(f"Entity type {entity_type} not enabled")

        # Extract fields
        value = entity_data.get("value", "")
        normalized_value = entity_data.get("normalized_value", value)
        confidence = float(entity_data.get("confidence", 0.0))

        # Create entity
        entity = Entity(
            id=str(uuid.uuid4()),
            type=entity_type,
            value=value,
            normalized_value=normalized_value,
            confidence=confidence,
            source_chunk_ids=[chunk_id],
            first_mention_chunk_id=chunk_id,
            document_id=document_id,
            section_path=section_path,
            metadata={
                "context": entity_data.get("context", ""),
                "extraction_confidence": confidence,
            },
            extraction_method="llm",
            extracted_at=datetime.now(),
        )

        return entity

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Deduplicate entities based on (type, normalized_value).

        Merges duplicate entities by:
        - Keeping highest confidence
        - Merging source_chunk_ids
        - Keeping first_mention_chunk_id from earliest chunk
        """
        # Group by (type, normalized_value)
        entity_groups: Dict[tuple, List[Entity]] = {}

        for entity in entities:
            key = (entity.type, entity.normalized_value)
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)

        # Merge duplicates
        deduplicated = []
        for key, group in entity_groups.items():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Merge multiple entities
                merged = self._merge_entities(group)
                deduplicated.append(merged)

        logger.info(f"Deduplicated {len(entities)} entities to {len(deduplicated)}")
        return deduplicated

    def _merge_entities(self, entities: List[Entity]) -> Entity:
        """Merge multiple entities into one."""
        # Sort by confidence (descending)
        entities = sorted(entities, key=lambda e: e.confidence, reverse=True)

        # Use highest-confidence entity as base
        merged = entities[0]

        # Merge source_chunk_ids from all entities
        all_chunk_ids = []
        for entity in entities:
            all_chunk_ids.extend(entity.source_chunk_ids)
        merged.source_chunk_ids = list(set(all_chunk_ids))  # Deduplicate

        # Keep first_mention_chunk_id from earliest extraction
        first_mentions = [e.first_mention_chunk_id for e in entities if e.first_mention_chunk_id]
        if first_mentions:
            merged.first_mention_chunk_id = first_mentions[0]

        # Merge metadata
        merged.metadata["merged_from"] = len(entities)
        merged.metadata["all_confidences"] = [e.confidence for e in entities]

        return merged

    def extract_from_text(self, text: str, chunk_id: Optional[str] = None) -> List[Entity]:
        """
        Extract entities from raw text (convenience method for testing).

        Args:
            text: Text content to extract from
            chunk_id: Optional chunk ID (auto-generated if not provided)

        Returns:
            List of extracted entities
        """
        chunk_id = chunk_id or str(uuid.uuid4())

        chunk = {
            "id": chunk_id,
            "raw_content": text,
            "content": text,
            "metadata": {},
        }

        return self._extract_from_single_chunk(chunk)
