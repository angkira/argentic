from typing import List, Dict, Any, Optional, Literal, Union
import time
import chromadb
from pydantic import BaseModel, Field

from argentic.core.protocol.message import BaseMessage
from argentic.core.messager.messager import Messager

DEFAULT_ENVIRONMENT_COLLECTION = "default_environment"


class Point3D(BaseModel):
    """Represents a 3D point with coordinates."""

    x: float
    y: float
    z: float

    def to_coordinate_string(self) -> str:
        """Convert to a coordinate string for storage."""
        return f"({self.x:.6f},{self.y:.6f},{self.z:.6f})"

    @classmethod
    def from_coordinate_string(cls, coord_str: str) -> "Point3D":
        """Parse coordinate string back to Point3D."""
        # Remove parentheses and split by comma
        coords = coord_str.strip("()").split(",")
        return cls(x=float(coords[0]), y=float(coords[1]), z=float(coords[2]))


class BoundingBox(BaseModel):
    """Represents a 3D bounding box."""

    min_point: Point3D
    max_point: Point3D

    def to_coordinate_string(self) -> str:
        """Convert bounding box to coordinate string."""
        return (
            f"bbox[{self.min_point.to_coordinate_string()}-{self.max_point.to_coordinate_string()}]"
        )

    @classmethod
    def from_coordinate_string(cls, coord_str: str) -> "BoundingBox":
        """Parse bounding box coordinate string."""
        # Remove 'bbox[' and ']' and split by '-'
        inner = coord_str.replace("bbox[", "").replace("]", "")
        min_str, max_str = inner.split("-")
        return cls(
            min_point=Point3D.from_coordinate_string(min_str),
            max_point=Point3D.from_coordinate_string(max_str),
        )

    def contains_point(self, point: Point3D) -> bool:
        """Check if a point is inside this bounding box."""
        return (
            self.min_point.x <= point.x <= self.max_point.x
            and self.min_point.y <= point.y <= self.max_point.y
            and self.min_point.z <= point.z <= self.max_point.z
        )

    def volume(self) -> float:
        """Calculate the volume of the bounding box."""
        return (
            (self.max_point.x - self.min_point.x)
            * (self.max_point.y - self.min_point.y)
            * (self.max_point.z - self.min_point.z)
        )


class EnvironmentEntry(BaseModel):
    """Represents an entry in the environment with either a point or bounding box."""

    object_id: str
    coordinates: Union[Point3D, BoundingBox]
    object_type: str  # e.g., "obstacle", "landmark", "target", etc.
    description: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_coordinate_string(self) -> str:
        """Get the coordinate representation for storage."""
        return self.coordinates.to_coordinate_string()

    def is_bounding_box(self) -> bool:
        """Check if this entry represents a bounding box."""
        return isinstance(self.coordinates, BoundingBox)


class AddEnvironmentEntryMessage(BaseMessage[None]):
    """Message for adding environment entries."""

    type: Literal["ADD_ENVIRONMENT_ENTRY"] = "ADD_ENVIRONMENT_ENTRY"
    entry: EnvironmentEntry
    collection_name: Optional[str] = None


class QueryEnvironmentMessage(BaseMessage[None]):
    """Message for querying environment entries."""

    type: Literal["QUERY_ENVIRONMENT"] = "QUERY_ENVIRONMENT"
    query_point: Optional[Point3D] = None
    query_box: Optional[BoundingBox] = None
    radius: Optional[float] = None
    object_type_filter: Optional[str] = None
    collection_name: Optional[str] = None


class EnvironmentManager:
    """Manages 3D coordinate-based storage using ChromaDB without embeddings."""

    def __init__(
        self,
        db_client: chromadb.ClientAPI,
        messager: Messager,
        default_collection_name: str = DEFAULT_ENVIRONMENT_COLLECTION,
    ):
        self.db_client = db_client
        self.messager = messager
        self.default_collection_name = default_collection_name
        self.collections: Dict[str, chromadb.Collection] = {}

    async def async_init(self):
        """Initialize the environment manager asynchronously."""
        try:
            await self.get_or_create_collection(self.default_collection_name)
            await self.messager.log(
                f"EnvironmentManager init complete. Default collection '{self.default_collection_name}' ready."
            )
        except Exception as e:
            await self.messager.log(
                f"Error during EnvironmentManager async_init: {e}", level="error"
            )
            raise

    async def get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection for environment storage."""
        if collection_name not in self.collections:
            await self.messager.log(f"Initializing environment collection: '{collection_name}'")
            try:
                # Create collection without embedding function (we'll use coordinates directly)
                collection = self.db_client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "3D coordinate-based environment storage"},
                )
                self.collections[collection_name] = collection
                await self.messager.log(
                    f"Environment collection '{collection_name}' created successfully."
                )
            except Exception as e:
                await self.messager.log(
                    f"Failed to create environment collection '{collection_name}': {e}",
                    level="error",
                )
                raise
        return self.collections[collection_name]

    async def store_entry(
        self,
        entry: EnvironmentEntry,
        collection_name: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Store an environment entry (point or bounding box) in the collection."""
        target_collection = collection_name or self.default_collection_name

        try:
            collection = await self.get_or_create_collection(target_collection)

            if timestamp is None:
                timestamp = time.time()

            # Prepare metadata
            metadata = {
                "object_id": entry.object_id,
                "object_type": entry.object_type,
                "description": entry.description,
                "confidence": entry.confidence,
                "timestamp": timestamp,
                "collection": target_collection,
                "coordinate_type": "bounding_box" if entry.is_bounding_box() else "point",
                **entry.metadata,
            }

            # Add coordinate information to metadata for filtering
            if isinstance(entry.coordinates, Point3D):
                metadata.update(
                    {
                        "x": entry.coordinates.x,
                        "y": entry.coordinates.y,
                        "z": entry.coordinates.z,
                    }
                )
            else:  # BoundingBox
                metadata.update(
                    {
                        "min_x": entry.coordinates.min_point.x,
                        "min_y": entry.coordinates.min_point.y,
                        "min_z": entry.coordinates.min_point.z,
                        "max_x": entry.coordinates.max_point.x,
                        "max_y": entry.coordinates.max_point.y,
                        "max_z": entry.coordinates.max_point.z,
                    }
                )

            # Use coordinate string as document content and unique ID
            coordinate_str = entry.get_coordinate_string()
            document_text = f"{entry.object_type}: {entry.description} at {coordinate_str}"

            # Store in ChromaDB - using add method with explicit ID
            collection.add(documents=[document_text], metadatas=[metadata], ids=[entry.object_id])

            await self.messager.log(
                f"Stored environment entry '{entry.object_id}' in collection '{target_collection}': {coordinate_str}"
            )
            return True

        except Exception as e:
            await self.messager.log(
                f"Error storing environment entry '{entry.object_id}': {e}", level="error"
            )
            return False

    async def query_by_point(
        self,
        point: Point3D,
        radius: Optional[float] = None,
        object_type_filter: Optional[str] = None,
        collection_name: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query environment entries near a specific point."""
        target_collection = collection_name or self.default_collection_name

        if target_collection not in self.collections:
            await self.messager.log(f"Collection '{target_collection}' not found.", level="warning")
            return []

        try:
            collection = self.collections[target_collection]

            # Build where clause for filtering
            where_clause: Optional[Dict[str, Any]] = None
            if object_type_filter:
                where_clause = {"object_type": {"$eq": object_type_filter}}

            # Get all documents with optional type filter
            results = collection.get(where=where_clause, include=["metadatas", "documents"])

            # Filter by distance if radius is specified
            filtered_results = []
            for i, metadata in enumerate(results["metadatas"] or []):
                if metadata.get("coordinate_type") == "point":
                    # Calculate distance for point entries
                    entry_point = Point3D(x=metadata["x"], y=metadata["y"], z=metadata["z"])
                    distance = self._calculate_distance(point, entry_point)

                    if radius is None or distance <= radius:
                        result_data = {
                            "object_id": metadata.get("object_id"),
                            "object_type": metadata.get("object_type"),
                            "description": metadata.get("description"),
                            "coordinates": entry_point,
                            "distance": distance,
                            "confidence": metadata.get("confidence", 1.0),
                            "metadata": {
                                k: v
                                for k, v in metadata.items()
                                if k
                                not in [
                                    "object_id",
                                    "object_type",
                                    "description",
                                    "x",
                                    "y",
                                    "z",
                                    "confidence",
                                    "timestamp",
                                    "collection",
                                    "coordinate_type",
                                ]
                            },
                            "timestamp": metadata.get("timestamp"),
                            "document": results["documents"][i] if results["documents"] else "",
                        }
                        filtered_results.append(result_data)

                elif metadata.get("coordinate_type") == "bounding_box":
                    # Check if point is inside bounding box
                    bbox = BoundingBox(
                        min_point=Point3D(
                            x=metadata["min_x"], y=metadata["min_y"], z=metadata["min_z"]
                        ),
                        max_point=Point3D(
                            x=metadata["max_x"], y=metadata["max_y"], z=metadata["max_z"]
                        ),
                    )

                    if bbox.contains_point(point):
                        result_data = {
                            "object_id": metadata.get("object_id"),
                            "object_type": metadata.get("object_type"),
                            "description": metadata.get("description"),
                            "coordinates": bbox,
                            "distance": 0.0,  # Inside bounding box
                            "confidence": metadata.get("confidence", 1.0),
                            "metadata": {
                                k: v
                                for k, v in metadata.items()
                                if k
                                not in [
                                    "object_id",
                                    "object_type",
                                    "description",
                                    "min_x",
                                    "min_y",
                                    "min_z",
                                    "max_x",
                                    "max_y",
                                    "max_z",
                                    "confidence",
                                    "timestamp",
                                    "collection",
                                    "coordinate_type",
                                ]
                            },
                            "timestamp": metadata.get("timestamp"),
                            "document": results["documents"][i] if results["documents"] else "",
                        }
                        filtered_results.append(result_data)

            # Sort by distance and limit results
            filtered_results.sort(key=lambda x: x["distance"])
            return filtered_results[:max_results]

        except Exception as e:
            await self.messager.log(f"Error querying environment by point: {e}", level="error")
            return []

    async def query_by_bounding_box(
        self,
        bbox: BoundingBox,
        object_type_filter: Optional[str] = None,
        collection_name: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query environment entries that intersect with a bounding box."""
        target_collection = collection_name or self.default_collection_name

        if target_collection not in self.collections:
            await self.messager.log(f"Collection '{target_collection}' not found.", level="warning")
            return []

        try:
            collection = self.collections[target_collection]

            # Build where clause for filtering
            where_clause: Optional[Dict[str, Any]] = None
            if object_type_filter:
                where_clause = {"object_type": {"$eq": object_type_filter}}

            # Get all documents with optional type filter
            results = collection.get(where=where_clause, include=["metadatas", "documents"])

            # Filter by bounding box intersection
            filtered_results = []
            for i, metadata in enumerate(results["metadatas"] or []):
                intersects = False
                coordinates = None

                if metadata.get("coordinate_type") == "point":
                    # Check if point is inside query bounding box
                    point = Point3D(x=metadata["x"], y=metadata["y"], z=metadata["z"])
                    if bbox.contains_point(point):
                        intersects = True
                        coordinates = point

                elif metadata.get("coordinate_type") == "bounding_box":
                    # Check bounding box intersection
                    entry_bbox = BoundingBox(
                        min_point=Point3D(
                            x=metadata["min_x"], y=metadata["min_y"], z=metadata["min_z"]
                        ),
                        max_point=Point3D(
                            x=metadata["max_x"], y=metadata["max_y"], z=metadata["max_z"]
                        ),
                    )

                    if self._bounding_boxes_intersect(bbox, entry_bbox):
                        intersects = True
                        coordinates = entry_bbox

                if intersects:
                    result_data = {
                        "object_id": metadata.get("object_id"),
                        "object_type": metadata.get("object_type"),
                        "description": metadata.get("description"),
                        "coordinates": coordinates,
                        "confidence": metadata.get("confidence", 1.0),
                        "metadata": {
                            k: v
                            for k, v in metadata.items()
                            if k
                            not in [
                                "object_id",
                                "object_type",
                                "description",
                                "x",
                                "y",
                                "z",
                                "min_x",
                                "min_y",
                                "min_z",
                                "max_x",
                                "max_y",
                                "max_z",
                                "confidence",
                                "timestamp",
                                "collection",
                                "coordinate_type",
                            ]
                        },
                        "timestamp": metadata.get("timestamp"),
                        "document": results["documents"][i] if results["documents"] else "",
                    }
                    filtered_results.append(result_data)

            return filtered_results[:max_results]

        except Exception as e:
            await self.messager.log(
                f"Error querying environment by bounding box: {e}", level="error"
            )
            return []

    async def remove_entry(self, object_id: str, collection_name: Optional[str] = None) -> bool:
        """Remove an environment entry by object ID."""
        target_collection = collection_name or self.default_collection_name

        if target_collection not in self.collections:
            await self.messager.log(f"Collection '{target_collection}' not found.", level="warning")
            return False

        try:
            collection = self.collections[target_collection]
            collection.delete(ids=[object_id])

            await self.messager.log(
                f"Removed environment entry '{object_id}' from collection '{target_collection}'"
            )
            return True

        except Exception as e:
            await self.messager.log(
                f"Error removing environment entry '{object_id}': {e}", level="error"
            )
            return False

    def _calculate_distance(self, point1: Point3D, point2: Point3D) -> float:
        """Calculate Euclidean distance between two 3D points."""
        return (
            (point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2
        ) ** 0.5

    def _bounding_boxes_intersect(self, bbox1: BoundingBox, bbox2: BoundingBox) -> bool:
        """Check if two bounding boxes intersect."""
        return (
            bbox1.min_point.x <= bbox2.max_point.x
            and bbox1.max_point.x >= bbox2.min_point.x
            and bbox1.min_point.y <= bbox2.max_point.y
            and bbox1.max_point.y >= bbox2.min_point.y
            and bbox1.min_point.z <= bbox2.max_point.z
            and bbox1.max_point.z >= bbox2.min_point.z
        )

    async def get_all_collections(self) -> List[str]:
        """Get list of all environment collections."""
        try:
            # Get all collections from ChromaDB
            all_collections = self.db_client.list_collections()
            env_collections = [
                col.name
                for col in all_collections
                if col.metadata
                and col.metadata.get("description") == "3D coordinate-based environment storage"
            ]
            return env_collections
        except Exception as e:
            await self.messager.log(f"Error listing environment collections: {e}", level="error")
            return []

    async def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a collection."""
        target_collection = collection_name or self.default_collection_name

        if target_collection not in self.collections:
            return {"error": f"Collection '{target_collection}' not found", "count": 0}

        try:
            collection = self.collections[target_collection]
            results = collection.get(include=["metadatas"])

            total_count = len(results["metadatas"] or [])
            point_count = sum(
                1 for m in (results["metadatas"] or []) if m.get("coordinate_type") == "point"
            )
            bbox_count = sum(
                1
                for m in (results["metadatas"] or [])
                if m.get("coordinate_type") == "bounding_box"
            )

            # Count by object type
            type_counts = {}
            for metadata in results["metadatas"] or []:
                obj_type = metadata.get("object_type", "unknown")
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

            return {
                "collection_name": target_collection,
                "total_count": total_count,
                "point_count": point_count,
                "bounding_box_count": bbox_count,
                "object_type_counts": type_counts,
            }

        except Exception as e:
            await self.messager.log(f"Error getting collection stats: {e}", level="error")
            return {"error": str(e), "count": 0}
