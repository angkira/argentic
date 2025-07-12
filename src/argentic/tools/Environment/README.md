# Environment Tool

The Environment Tool provides 3D spatial data storage and querying capabilities for the Argentic framework. It stores objects and their spatial information using coordinates rather than embeddings, making it ideal for visual perception, navigation, and spatial reasoning tasks.

## Features

### Coordinate Types

- **Point3D**: Single 3D points with (x, y, z) coordinates
- **BoundingBox**: 3D bounding boxes defined by min and max points

### Storage Capabilities

- Store objects with either point coordinates or bounding boxes
- Associate metadata and descriptions with spatial objects
- Confidence scores for object detection reliability
- Object type categorization (obstacles, landmarks, targets, etc.)

### Query Operations

- **Point Queries**: Find objects near a specific 3D point within a radius
- **Bounding Box Queries**: Find objects that intersect with a 3D region
- **Filtered Queries**: Filter results by object type
- **Distance-based Sorting**: Results sorted by proximity

## Usage

### Actions

#### Store Objects

```python
# Store a point object
action = "store"
object_id = "landmark_001"
object_type = "landmark"
description = "Main entrance door"
x, y, z = 10.5, 20.3, 1.8
confidence = 0.95

# Store a bounding box object
action = "store"
object_id = "obstacle_002"
object_type = "obstacle"
description = "Large table"
min_x, min_y, min_z = 5.0, 10.0, 0.0
max_x, max_y, max_z = 8.0, 12.0, 1.2
```

#### Query Near Point

```python
action = "query_point"
x, y, z = 10.0, 20.0, 2.0
radius = 5.0
object_type_filter = "obstacle"  # optional
max_results = 10  # optional
```

#### Query Bounding Box Intersection

```python
action = "query_bbox"
min_x, min_y, min_z = 0.0, 0.0, 0.0
max_x, max_y, max_z = 20.0, 30.0, 3.0
object_type_filter = "landmark"  # optional
```

#### Remove Objects

```python
action = "remove"
object_id = "obstacle_002"
```

#### Get Statistics

```python
action = "stats"
collection_name = "visual_perception"  # optional
```

#### List Collections

```python
action = "list_collections"
```

## Configuration

The tool is configured via `environment_config.yaml`:

```yaml
environment_storage:
  base_directory: ./environment_db
  default_collection: spatial_environment

collections:
  spatial_environment:
    max_entries: 10000
    max_distance_query: 100.0

  visual_perception:
    max_entries: 5000
    max_distance_query: 50.0

default_query:
  max_results: 10
  default_radius: 5.0

coordinate_system:
  min_x: -1000.0
  max_x: 1000.0
  min_y: -1000.0
  max_y: 1000.0
  min_z: -100.0
  max_z: 100.0
  coordinate_precision: 6
```

## Use Cases

### Visual Perception

- Store detected objects from camera/sensor data
- Query objects in the robot's field of view
- Track object movements and changes

### Navigation

- Store landmarks for localization
- Mark obstacles for path planning
- Define safe/unsafe zones

### Manipulation

- Store target object locations
- Track workspace boundaries
- Remember successful grasp points

### Spatial Reasoning

- Analyze spatial relationships between objects
- Plan movements in 3D space
- Understand environment layout

## Data Structure

### Environment Entry

```python
{
    "object_id": "unique_identifier",
    "object_type": "obstacle|landmark|target|...",
    "description": "Human readable description",
    "coordinates": Point3D | BoundingBox,
    "confidence": 0.95,
    "metadata": {
        "color": "red",
        "material": "plastic",
        "detected_by": "camera_1"
    }
}
```

### Query Results

```python
{
    "object_id": "landmark_001",
    "object_type": "landmark",
    "description": "Main entrance door",
    "coordinates": Point3D(10.5, 20.3, 1.8),
    "distance": 2.1,
    "confidence": 0.95,
    "timestamp": 1703123456.789,
    "metadata": {...}
}
```

## Running the Service

Start the environment tool service:

```bash
python src/services/environment_tool_service.py
```

The service will:

1. Initialize ChromaDB for coordinate storage
2. Connect to the messaging system (MQTT)
3. Register the environment_tool with the agent
4. Listen for spatial data operations

## Integration

The Environment Tool integrates with:

- **Visual Perception Systems**: Store detected objects
- **Navigation Systems**: Query obstacles and landmarks
- **Manipulation Systems**: Track target objects
- **Planning Systems**: Reason about spatial constraints
- **Mapping Systems**: Build environmental representations
