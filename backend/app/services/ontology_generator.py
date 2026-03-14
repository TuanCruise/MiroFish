"""
Ontology Generation Service
Interface 1: Analyze text content, generate entity and relationship type definitions suitable for social simulation
"""

import json
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient


# Ontology generation system prompt
ONTOLOGY_SYSTEM_PROMPT = """You are a schema designer for a social media simulation platform. Given source material and a simulation goal, design actor categories and interaction types for a public opinion simulation.

Return ONLY a data object. No explanations. No prose. The object needs three top-level keys: entity_types (list), edge_types (list), analysis_summary (string).

For entity_types, provide exactly 10 entries. Last 2 must be Person and Organization as catch-all fallbacks. First 8 are specific types from the text context. Entities must be real-world actors (people, companies, agencies, media) that can interact on social media. NOT abstract concepts like opinions or trends. Use full_name, org_name, title, role, position as field names (avoid: name, uuid, group_id, created_at, summary).

For edge_types, provide 6-10 entries with UPPER_SNAKE_CASE names describing real social connections. Match source and target to your entity type names.

Suggested actor categories: Student, Professor, Journalist, Celebrity, Executive, Official, Lawyer, Doctor, University, Company, GovernmentAgency, MediaOutlet, Hospital, School, NGO
Suggested interaction types: WORKS_FOR, STUDIES_AT, AFFILIATED_WITH, REPRESENTS, REGULATES, REPORTS_ON, COMMENTS_ON, RESPONDS_TO, SUPPORTS, OPPOSES, COLLABORATES_WITH, COMPETES_WITH
"""


class OntologyGenerator:
    """
    Ontology Generator
    Analyze text content, generate entity and relationship type definitions
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate ontology definition
        
        Args:
            document_texts: List of document texts
            simulation_requirement: Simulation requirement description
            additional_context: Additional context
            
        Returns:
            Ontology definition (entity_types, edge_types, etc.)
        """
        # Build user message
        user_message = self._build_user_message(
            document_texts, 
            simulation_requirement,
            additional_context
        )
        
        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # Call LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )
        
        # Validate and post-process
        result = self._validate_and_process(result)
        
        return result
    
    # Max text length for LLM (50,000 characters)
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    
    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Build user message"""
        
        # Merge texts
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)
        
        # If text exceeds char limits
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(Original was {original_length} chars, truncated to first {self.MAX_TEXT_LENGTH_FOR_LLM} chars for ontology analysis)..."
        
        message = f"""## Simulation Requirements

{simulation_requirement}

## Document Content

{combined_text}
"""
        
        if additional_context:
            message += f"""
## Additional Context

{additional_context}
"""
        
        message += """
Please design entity types and relationship types suitable for social public opinion simulation based on the above content.

**Rules that must be obeyed**:
1. You must output EXACTLY 10 entity types.
2. The last 2 must be fallback types: Person (individual fallback) and Organization (organization fallback).
3. The first 8 are specific types designed based on the text content.
4. All entity types must be subjects that can speak in reality, not abstract concepts.
5. Attribute names cannot use reserved words like name, uuid, group_id. Use full_name, org_name, etc. instead.
"""
        
        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and post-process results"""
        
        # Ensure required fields exist
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        
        # Validate entity types
        normalized_entities = []
        for entity in result["entity_types"]:
            # Normalize field names: Gemini may return 'type_name' instead of 'name'
            if "name" not in entity:
                entity["name"] = entity.pop("type_name", entity.pop("entity_name", entity.pop("type", None)))
            if not entity.get("name"):
                continue  # Skip entities with no name
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # Ensure description does not exceed 100 characters
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
            normalized_entities.append(entity)
        result["entity_types"] = normalized_entities
        
        # Validate relationship types
        normalized_edges = []
        for edge in result["edge_types"]:
            # Normalize field names: Gemini may return 'edge_name' or 'relationship_name'
            if "name" not in edge:
                edge["name"] = edge.pop("edge_name", edge.pop("relationship_name", edge.pop("type_name", None)))
            if not edge.get("name"):
                continue  # Skip edges with no name
            # Normalize source_targets: Gemini may use different field name
            if "source_targets" not in edge:
                edge["source_targets"] = edge.pop("source_target", edge.pop("connections", []))
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
            normalized_edges.append(edge)
        result["edge_types"] = normalized_edges
        
        # Zep API limit: max 10 custom entity types, max 10 custom edge types
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10
        
        # Fallback type definitions
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }
        
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        # Check if fallback types already exist
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names
        
        # Fallback types to add
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)
        
        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)
            
            # If adding them exceeds 10, need to remove some existing types
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Calculate how many to remove
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Remove from the end (keep the more important specific types at the beginning)
                result["entity_types"] = result["entity_types"][:-to_remove]
            
            # Add fallback types
            result["entity_types"].extend(fallbacks_to_add)
        
        # Finally ensure the limit is not exceeded (defensive programming)
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]
        
        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        Convert ontology definition to Python code (similar to ontology.py)
        
        Args:
            ontology: Ontology definition
            
        Returns:
            Python code string
        """
        code_lines = [
            '"""',
            'Custom entity type definitions',
            'Automatically generated by MiroFish for social public opinion simulation',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== Entity Type Definitions ==============',
            '',
        ]
        
        # Generate entity types
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== Relationship Type Definitions ==============')
        code_lines.append('')
        
        # Generate relationship types
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # Convert to PascalCase class name
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        # Generate type dictionaries
        code_lines.append('# ============== Type Configurations ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        # Generate source_targets mapping for edges
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)

