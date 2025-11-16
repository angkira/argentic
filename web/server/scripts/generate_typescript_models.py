#!/usr/bin/env python3
"""Generate TypeScript interfaces from FastAPI/Pydantic models."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# Add parent directory to path to import app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app


def python_type_to_typescript(schema: Dict[str, Any], definitions: Dict[str, Any]) -> str:
    """Convert Python/JSON Schema type to TypeScript type."""
    if "$ref" in schema:
        # Reference to another model
        ref_name = schema["$ref"].split("/")[-1]
        return ref_name

    schema_type = schema.get("type")
    schema_format = schema.get("format")

    if schema_type == "string":
        if "enum" in schema:
            # Enum type
            enum_values = schema["enum"]
            return " | ".join(f"'{v}'" for v in enum_values)
        if schema_format == "date-time":
            return "string"  # Could use Date, but string is safer for JSON
        return "string"
    elif schema_type == "integer":
        return "number"
    elif schema_type == "number":
        return "number"
    elif schema_type == "boolean":
        return "boolean"
    elif schema_type == "array":
        items_type = python_type_to_typescript(schema.get("items", {}), definitions)
        return f"{items_type}[]"
    elif schema_type == "object":
        if "additionalProperties" in schema:
            value_type = python_type_to_typescript(
                schema.get("additionalProperties", {}), definitions
            )
            return f"Record<string, {value_type}>"
        return "Record<string, any>"
    elif schema_type == "null":
        return "null"

    # Handle anyOf, oneOf, allOf
    if "anyOf" in schema:
        types = [python_type_to_typescript(s, definitions) for s in schema["anyOf"]]
        return " | ".join(types)
    if "oneOf" in schema:
        types = [python_type_to_typescript(s, definitions) for s in schema["oneOf"]]
        return " | ".join(types)

    return "any"


def generate_interface(
    name: str, schema: Dict[str, Any], definitions: Dict[str, Any]
) -> str:
    """Generate a TypeScript interface from a JSON schema."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    lines = [f"export interface {name} {{"]

    for prop_name, prop_schema in properties.items():
        ts_type = python_type_to_typescript(prop_schema, definitions)
        optional = "" if prop_name in required else "?"
        description = prop_schema.get("description", "")

        if description:
            lines.append(f"  /** {description} */")
        lines.append(f"  {prop_name}{optional}: {ts_type};")

    lines.append("}")

    return "\n".join(lines)


def generate_type_alias(name: str, schema: Dict[str, Any], definitions: Dict[str, Any]) -> str:
    """Generate a TypeScript type alias from a JSON schema."""
    if "enum" in schema:
        # Enum type
        enum_values = schema["enum"]
        values = " | ".join(f"'{v}'" for v in enum_values)
        return f"export type {name} = {values};"

    # For other types, convert to interface
    return generate_interface(name, schema, definitions)


def should_skip_model(name: str) -> bool:
    """Check if model should be skipped from generation."""
    skip_patterns = ["HTTPValidationError", "ValidationError", "Body_", "HTTPException"]
    return any(pattern in name for pattern in skip_patterns)


def generate_typescript_models(output_file: Path) -> None:
    """Generate TypeScript models from FastAPI OpenAPI schema."""
    # Get OpenAPI schema
    openapi_schema = app.openapi()

    # Extract component schemas
    schemas = openapi_schema.get("components", {}).get("schemas", {})

    # Track dependencies
    all_interfaces: List[str] = []
    processed: Set[str] = set()

    # Process each schema
    for schema_name, schema_def in sorted(schemas.items()):
        if should_skip_model(schema_name):
            continue

        if schema_name in processed:
            continue

        processed.add(schema_name)

        # Generate interface or type
        if "enum" in schema_def:
            ts_code = generate_type_alias(schema_name, schema_def, schemas)
        else:
            ts_code = generate_interface(schema_name, schema_def, schemas)

        all_interfaces.append(ts_code)

    # Write to file
    output = [
        "/* Auto-generated TypeScript interfaces from Python models */",
        "/* Do not edit manually - regenerate using: python scripts/generate_typescript_models.py */",
        "",
        *all_interfaces,
        "",
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(output))

    print(f"✓ Generated TypeScript models at: {output_file}")
    print(f"✓ Generated {len(all_interfaces)} interfaces/types")


def main():
    """Main entry point."""
    # Output to client models directory
    output_file = Path(__file__).parent.parent.parent / "client" / "src" / "app" / "models" / "generated.ts"

    try:
        generate_typescript_models(output_file)
        print("\n✨ TypeScript model generation complete!")
    except Exception as e:
        print(f"❌ Error generating TypeScript models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
