#!/usr/bin/env python
"""Helper script to fix JSON serialization issues with TestMessage classes."""

import json
import uuid
from datetime import datetime


def patch_json_module():
    """Patch the json module with a custom encoder that handles common types."""

    class UniversalEncoder(json.JSONEncoder):
        def default(self, obj):
            # Handle datetime objects
            if isinstance(obj, datetime):
                return obj.isoformat()

            # Handle UUID objects
            if isinstance(obj, uuid.UUID):
                return str(obj)

            # Try handling objects with __dict__
            if hasattr(obj, "__dict__"):
                # Try to extract TestMessage-like objects
                data = {}
                # Get all non-private attributes
                for key, value in obj.__dict__.items():
                    if not key.startswith("_"):
                        data[key] = value

                # Add special handling for common attributes
                if hasattr(obj, "id") and obj.id:
                    data["id"] = str(obj.id)
                if hasattr(obj, "timestamp") and obj.timestamp:
                    data["timestamp"] = (
                        obj.timestamp.isoformat()
                        if hasattr(obj.timestamp, "isoformat")
                        else str(obj.timestamp)
                    )
                if hasattr(obj, "__class__") and hasattr(obj.__class__, "__name__"):
                    data["type"] = obj.__class__.__name__

                return data

            # Try iterator objects
            if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
                try:
                    return dict(obj)
                except (TypeError, ValueError):
                    try:
                        return list(obj)
                    except (TypeError, ValueError):
                        pass

            # Let the base class handle it
            return super().default(obj)

    # Patch json.dumps to use our encoder by default
    original_dumps = json.dumps

    def patched_dumps(obj, *args, **kwargs):
        if "cls" not in kwargs:
            kwargs["cls"] = UniversalEncoder
        return original_dumps(obj, *args, **kwargs)

    json.dumps = patched_dumps
    json._default_encoder = UniversalEncoder()

    print("Applied universal JSON encoder")


def main():
    """Main function to fix serialization issues."""
    print("Applying JSON serialization fixes...")
    patch_json_module()

    print("Serialization fixes applied successfully")


if __name__ == "__main__":
    main()
