#!/usr/bin/env python3
"""
Lightweight TypeScript model generator.
Reads Pydantic models directly without needing full app dependencies.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set
import importlib.util

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_module_from_file(module_name: str, file_path: Path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def python_type_to_typescript(schema: Dict[str, Any], definitions: Dict[str, Any]) -> str:
    """Convert Python/JSON Schema type to TypeScript type."""
    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        return ref_name

    schema_type = schema.get("type")
    schema_format = schema.get("format")

    if schema_type == "string":
        if "enum" in schema:
            enum_values = schema["enum"]
            return " | ".join(f"'{v}'" for v in enum_values)
        if schema_format == "date-time":
            return "string"
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

    if "anyOf" in schema:
        types = [python_type_to_typescript(s, definitions) for s in schema["anyOf"]]
        return " | ".join(types)
    if "oneOf" in schema:
        types = [python_type_to_typescript(s, definitions) for s in schema["oneOf"]]
        return " | ".join(types)

    return "any"


def generate_interface(name: str, schema: Dict[str, Any], definitions: Dict[str, Any]) -> str:
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

        readonly_prefix = "readonly " if name.endswith("Response") or name.endswith("Info") else ""
        lines.append(f"  {readonly_prefix}{prop_name}{optional}: {ts_type};")

    lines.append("}")
    return "\n".join(lines)


def generate_typescript_models_simple(output_file: Path) -> None:
    """Generate TypeScript models by creating minimal FastAPI app."""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError:
        print("Error: FastAPI and Pydantic are required")
        print("Install with: uv sync")
        sys.exit(1)

    # Import model modules
    models_dir = Path(__file__).parent.parent / "app" / "models"

    # Create a minimal FastAPI app
    app = FastAPI()

    # Import all model files
    model_files = ["agent", "supervisor", "workflow", "llm", "messaging", "tool"]

    for model_file in model_files:
        try:
            module_path = models_dir / f"{model_file}.py"
            if module_path.exists():
                load_module_from_file(f"app.models.{model_file}", module_path)
        except Exception as e:
            print(f"Warning: Could not load {model_file}: {e}")

    # Import routes to register models with FastAPI
    try:
        from app.routes import agents, supervisors, workflows, config as config_routes
        app.include_router(agents.router)
        app.include_router(supervisors.router)
        app.include_router(workflows.router)
        app.include_router(config_routes.router)
    except Exception as e:
        print(f"Warning: Could not import routes: {e}")

    # Get OpenAPI schema
    try:
        openapi_schema = app.openapi()
    except Exception as e:
        print(f"Error generating OpenAPI schema: {e}")
        sys.exit(1)

    # Extract schemas
    schemas = openapi_schema.get("components", {}).get("schemas", {})

    # Generate interfaces
    all_interfaces: List[str] = []
    skip_patterns = ["HTTPValidationError", "ValidationError", "Body_", "HTTPException"]

    for schema_name, schema_def in sorted(schemas.items()):
        if any(pattern in schema_name for pattern in skip_patterns):
            continue

        if "enum" in schema_def:
            enum_values = schema_def["enum"]
            values = " | ".join(f"'{v}'" for v in enum_values)
            ts_code = f"export type {schema_name} = {values};"
        else:
            ts_code = generate_interface(schema_name, schema_def, schemas)

        all_interfaces.append(ts_code)

    # Write output
    output = [
        "/* eslint-disable */",
        "/**",
        " * Auto-generated TypeScript interfaces from Python Pydantic models",
        " * ",
        " * DO NOT EDIT MANUALLY!",
        " * ",
        " * To regenerate:",
        " *   cd web/client && npm run generate-models",
        " */",
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
    output_file = (
        Path(__file__).parent.parent.parent
        / "client"
        / "src"
        / "app"
        / "models"
        / "generated.ts"
    )

    try:
        generate_typescript_models_simple(output_file)
        print("\n✨ TypeScript model generation complete!")
    except Exception as e:
        print(f"❌ Error generating TypeScript models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
