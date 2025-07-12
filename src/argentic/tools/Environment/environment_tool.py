import time
import json
from typing import List, Dict, Any, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field
from argentic.tools.Environment.environment import (
    EnvironmentManager,
    EnvironmentEntry,
    Point3D,
    BoundingBox,
)
from argentic.core.messager.messager import Messager
from argentic.core.tools.tool_base import BaseTool
from argentic.core.logger import get_logger, LogLevel, parse_log_level


# --- Argument Schema --- Define actions
class EnvAction(str, Enum):
    STORE = "store"  # Store a new environment entry
    QUERY_POINT = "query_point"  # Query entries near a point
    QUERY_BBOX = "query_bbox"  # Query entries intersecting with a bounding box
    REMOVE = "remove"  # Remove an entry by object_id
    STATS = "stats"  # Get collection statistics
    LIST_COLLECTIONS = "list_collections"  # List available collections


class EnvironmentToolInput(BaseModel):
    action: EnvAction = Field(
        description="The action to perform: store (add entry), query_point (search near point), query_bbox (search in area), remove (delete entry), stats (get statistics), list_collections (get available collections)."
    )
    # Store action parameters
    object_id: Optional[str] = Field(
        None, description="Unique identifier for the object (required for store and remove actions)"
    )
    object_type: Optional[str] = Field(
        None, description="Type of object (e.g., 'obstacle', 'landmark', 'target')"
    )
    description: Optional[str] = Field(None, description="Description of the object")
    confidence: Optional[float] = Field(None, description="Confidence score between 0.0 and 1.0")

    # Coordinate parameters - for single point
    x: Optional[float] = Field(None, description="X coordinate")
    y: Optional[float] = Field(None, description="Y coordinate")
    z: Optional[float] = Field(None, description="Z coordinate")

    # Bounding box parameters
    min_x: Optional[float] = Field(None, description="Minimum X coordinate for bounding box")
    min_y: Optional[float] = Field(None, description="Minimum Y coordinate for bounding box")
    min_z: Optional[float] = Field(None, description="Minimum Z coordinate for bounding box")
    max_x: Optional[float] = Field(None, description="Maximum X coordinate for bounding box")
    max_y: Optional[float] = Field(None, description="Maximum Y coordinate for bounding box")
    max_z: Optional[float] = Field(None, description="Maximum Z coordinate for bounding box")

    # Query parameters
    radius: Optional[float] = Field(None, description="Search radius for point queries")
    object_type_filter: Optional[str] = Field(None, description="Filter results by object type")
    max_results: Optional[int] = Field(None, description="Maximum number of results to return")

    # Collection parameter
    collection_name: Optional[str] = Field(
        None, description="Optional name of specific collection to use"
    )

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata for the object"
    )

    def model_post_init(self, __context):
        if self.action == EnvAction.STORE:
            if not self.object_id:
                raise ValueError("'object_id' field is required when action is 'store'")
            if not self.object_type:
                raise ValueError("'object_type' field is required when action is 'store'")
            if not self.description:
                raise ValueError("'description' field is required when action is 'store'")
            # Check that we have either point coordinates or bounding box coordinates
            has_point = all(coord is not None for coord in [self.x, self.y, self.z])
            has_bbox = all(
                coord is not None
                for coord in [
                    self.min_x,
                    self.min_y,
                    self.min_z,
                    self.max_x,
                    self.max_y,
                    self.max_z,
                ]
            )
            if not has_point and not has_bbox:
                raise ValueError(
                    "Either point coordinates (x, y, z) or bounding box coordinates (min_x, min_y, min_z, max_x, max_y, max_z) are required for 'store' action"
                )
        elif self.action == EnvAction.QUERY_POINT:
            if not all(coord is not None for coord in [self.x, self.y, self.z]):
                raise ValueError(
                    "Point coordinates (x, y, z) are required when action is 'query_point'"
                )
        elif self.action == EnvAction.QUERY_BBOX:
            if not all(
                coord is not None
                for coord in [
                    self.min_x,
                    self.min_y,
                    self.min_z,
                    self.max_x,
                    self.max_y,
                    self.max_z,
                ]
            ):
                raise ValueError(
                    "Bounding box coordinates (min_x, min_y, min_z, max_x, max_y, max_z) are required when action is 'query_bbox'"
                )
        elif self.action == EnvAction.REMOVE:
            if not self.object_id:
                raise ValueError("'object_id' field is required when action is 'remove'")


def format_environment_results(results: List[Dict[str, Any]]) -> str:
    """Format environment query results for the LLM."""
    if not results:
        return "No environment entries found for the query."

    formatted_results = []
    for result in results:
        coord_info = ""
        coordinates = result.get("coordinates")
        if isinstance(coordinates, Point3D):
            coord_info = f"Point({coordinates.x:.2f}, {coordinates.y:.2f}, {coordinates.z:.2f})"
        elif isinstance(coordinates, BoundingBox):
            coord_info = f"BBox[({coordinates.min_point.x:.2f}, {coordinates.min_point.y:.2f}, {coordinates.min_point.z:.2f}) - ({coordinates.max_point.x:.2f}, {coordinates.max_point.y:.2f}, {coordinates.max_point.z:.2f})]"

        distance_info = ""
        if "distance" in result:
            distance_info = f", Distance: {result['distance']:.2f}"

        timestamp_info = ""
        if result.get("timestamp"):
            try:
                ts_str = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(float(result["timestamp"]))
                )
                timestamp_info = f", Time: {ts_str}"
            except (ValueError, TypeError):
                timestamp_info = f", Time: {result['timestamp']}"

        confidence_info = ""
        if result.get("confidence") is not None:
            confidence_info = f", Confidence: {result['confidence']:.2f}"

        formatted_results.append(
            f"ID: {result.get('object_id', 'unknown')}, "
            f"Type: {result.get('object_type', 'unknown')}, "
            f"Description: {result.get('description', 'no description')}, "
            f"Location: {coord_info}{distance_info}{confidence_info}{timestamp_info}"
        )

    return "\n---\n".join(formatted_results)


class EnvironmentTool(BaseTool):
    """Tool for managing 3D spatial environment data."""

    def __init__(
        self,
        messager: Messager,
        environment_manager: Optional[EnvironmentManager] = None,
        log_level: Union[LogLevel, str] = LogLevel.INFO,
    ):
        """environment_manager is optional when instantiating in Agent for prompt-only tools."""
        # Build JSON API schema for tool registration
        api_schema = EnvironmentToolInput.model_json_schema()
        super().__init__(
            name="environment_tool",
            manual=(
                "Manages 3D spatial environment data. Use this tool to store and query objects in 3D space. "
                "Use 'store' to add new objects with either point coordinates or bounding boxes. "
                "Use 'query_point' to find objects near a specific 3D point within a radius. "
                "Use 'query_bbox' to find objects that intersect with a 3D bounding box. "
                "Use 'remove' to delete objects by their ID. "
                "Use 'stats' to get statistics about a collection. "
                "Use 'list_collections' to get all available environment collections. "
                "Objects can include obstacles, landmarks, targets, or any spatial entities with metadata."
            ),
            api=json.dumps(api_schema),
            argument_schema=EnvironmentToolInput,
            messager=messager,
        )
        self.environment_manager = environment_manager

        # Set up logger
        if isinstance(log_level, str):
            self.log_level = parse_log_level(log_level)
        else:
            self.log_level = log_level

        self.logger = get_logger("env_tool", self.log_level)
        self.logger.info("EnvironmentTool instance created")

        # Initialize handlers map
        self._action_handlers = {
            EnvAction.STORE: self._handle_store,
            EnvAction.QUERY_POINT: self._handle_query_point,
            EnvAction.QUERY_BBOX: self._handle_query_bbox,
            EnvAction.REMOVE: self._handle_remove,
            EnvAction.STATS: self._handle_stats,
            EnvAction.LIST_COLLECTIONS: self._handle_list_collections,
        }

    def set_log_level(self, level: Union[LogLevel, str]) -> None:
        """Set the log level for the tool"""
        if isinstance(level, str):
            self.log_level = parse_log_level(level)
        else:
            self.log_level = level

        self.logger.setLevel(self.log_level.value)
        self.logger.info(f"Log level changed to {self.log_level.name}")

        # Update handlers
        for handler in self.logger.handlers:
            handler.setLevel(self.log_level.value)

    async def _execute(self, **kwargs) -> Any:
        """Execute the requested action on the environment."""
        action = kwargs.get("action")
        self.logger.info(f"Executing action: {action}")

        # Check if environment_manager is available
        if self.environment_manager is None:
            error_msg = (
                "Environment manager is not available. Cannot execute environment operations."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Get handler for the action
        handler = self._action_handlers.get(action)
        if handler is None:
            error_msg = f"Unsupported action: {action}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Execute the handler
        return await handler(**kwargs)

    async def _handle_store(self, **kwargs) -> str:
        """Handle store action."""
        if self.environment_manager is None:
            raise RuntimeError("Environment manager is not available")

        object_id = kwargs.get("object_id")
        object_type = kwargs.get("object_type")
        description = kwargs.get("description")
        confidence = kwargs.get("confidence")
        x = kwargs.get("x")
        y = kwargs.get("y")
        z = kwargs.get("z")
        min_x = kwargs.get("min_x")
        min_y = kwargs.get("min_y")
        min_z = kwargs.get("min_z")
        max_x = kwargs.get("max_x")
        max_y = kwargs.get("max_y")
        max_z = kwargs.get("max_z")
        collection_name = kwargs.get("collection_name")
        metadata = kwargs.get("metadata")

        # Create coordinates object
        coordinates = None
        if all(coord is not None for coord in [x, y, z]):
            coordinates = Point3D(x=x, y=y, z=z)
        elif all(coord is not None for coord in [min_x, min_y, min_z, max_x, max_y, max_z]):
            coordinates = BoundingBox(
                min_point=Point3D(x=min_x, y=min_y, z=min_z),
                max_point=Point3D(x=max_x, y=max_y, z=max_z),
            )
        else:
            error_msg = "Invalid coordinates provided for store action"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Create environment entry
        entry = EnvironmentEntry(
            object_id=object_id or "",  # Provide empty string as fallback
            coordinates=coordinates,
            object_type=object_type or "",
            description=description or "",
            confidence=confidence or 1.0,
            metadata=metadata or {},
        )

        self.logger.info(
            f"Storing entry '{object_id}' of type '{object_type}' in collection '{collection_name or 'default'}'"
        )
        success = await self.environment_manager.store_entry(
            entry=entry, collection_name=collection_name
        )

        result_msg = f"Store action {'succeeded' if success else 'failed'} for object '{object_id}' in collection '{collection_name or 'default'}'."
        self.logger.info(result_msg)
        return result_msg

    async def _handle_query_point(self, **kwargs) -> str:
        """Handle query_point action."""
        if self.environment_manager is None:
            raise RuntimeError("Environment manager is not available")

        x = kwargs.get("x")
        y = kwargs.get("y")
        z = kwargs.get("z")
        radius = kwargs.get("radius")
        object_type_filter = kwargs.get("object_type_filter")
        collection_name = kwargs.get("collection_name")
        max_results = kwargs.get("max_results")

        query_point = Point3D(x=x, y=y, z=z)
        self.logger.info(
            f"Querying near point ({x}, {y}, {z}) with radius {radius} in collection '{collection_name or 'default'}'"
        )

        results = await self.environment_manager.query_by_point(
            point=query_point,
            radius=radius,
            object_type_filter=object_type_filter,
            collection_name=collection_name,
            max_results=max_results or 10,
        )

        formatted_result = format_environment_results(results)
        self.logger.info(f"Found {len(results)} entries near point")
        return formatted_result

    async def _handle_query_bbox(self, **kwargs) -> str:
        """Handle query_bbox action."""
        if self.environment_manager is None:
            raise RuntimeError("Environment manager is not available")

        min_x = kwargs.get("min_x")
        min_y = kwargs.get("min_y")
        min_z = kwargs.get("min_z")
        max_x = kwargs.get("max_x")
        max_y = kwargs.get("max_y")
        max_z = kwargs.get("max_z")
        object_type_filter = kwargs.get("object_type_filter")
        collection_name = kwargs.get("collection_name")
        max_results = kwargs.get("max_results")

        query_bbox = BoundingBox(
            min_point=Point3D(x=min_x, y=min_y, z=min_z),
            max_point=Point3D(x=max_x, y=max_y, z=max_z),
        )
        self.logger.info(
            f"Querying bounding box from ({min_x}, {min_y}, {min_z}) to ({max_x}, {max_y}, {max_z}) in collection '{collection_name or 'default'}'"
        )

        results = await self.environment_manager.query_by_bounding_box(
            bbox=query_bbox,
            object_type_filter=object_type_filter,
            collection_name=collection_name,
            max_results=max_results or 10,
        )

        formatted_result = format_environment_results(results)
        self.logger.info(f"Found {len(results)} entries in bounding box")
        return formatted_result

    async def _handle_remove(self, **kwargs) -> str:
        """Handle remove action."""
        if self.environment_manager is None:
            raise RuntimeError("Environment manager is not available")

        object_id = kwargs.get("object_id")
        collection_name = kwargs.get("collection_name")

        if not object_id:
            error_msg = "object_id is required for remove action"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(
            f"Removing entry '{object_id}' from collection '{collection_name or 'default'}'"
        )
        success = await self.environment_manager.remove_entry(
            object_id=object_id, collection_name=collection_name
        )

        result_msg = f"Remove action {'succeeded' if success else 'failed'} for object '{object_id}' in collection '{collection_name or 'default'}'."
        self.logger.info(result_msg)
        return result_msg

    async def _handle_stats(self, **kwargs) -> str:
        """Handle stats action."""
        if self.environment_manager is None:
            raise RuntimeError("Environment manager is not available")

        collection_name = kwargs.get("collection_name")

        self.logger.info(f"Getting statistics for collection '{collection_name or 'default'}'")
        stats = await self.environment_manager.get_collection_stats(collection_name)
        self.logger.info(f"Collection stats retrieved: {stats.get('total_count', 0)} total entries")
        return json.dumps(stats, indent=2)

    async def _handle_list_collections(self, **kwargs) -> str:
        """Handle list_collections action."""
        if self.environment_manager is None:
            raise RuntimeError("Environment manager is not available")

        self.logger.info("Listing available environment collections")
        try:
            collections = await self.environment_manager.get_all_collections()
            result = {
                "collections": collections,
                "default_collection": self.environment_manager.default_collection_name,
                "count": len(collections),
            }
            self.logger.info(f"Found {len(collections)} environment collections")
            return json.dumps(result, indent=2)
        except Exception as e:
            error_msg = f"Error listing environment collections: {str(e)}"
            self.logger.error(error_msg)
            return json.dumps(
                {"collections": [], "default_collection": None, "error": error_msg, "count": 0},
                indent=2,
            )
