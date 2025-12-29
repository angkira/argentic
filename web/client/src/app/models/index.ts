/**
 * Model exports - Hybrid approach
 *
 * Option 1: Use auto-generated models from Python backend
 * Uncomment the line below after running: npm run generate-models
 */
// export * from './generated';

/**
 * Option 2: Use manual models (current)
 * These are kept in sync manually with Python Pydantic models
 */
export * from './agent.model';
export * from './supervisor.model';
export * from './workflow.model';
export * from './config.model';

/**
 * Frontend-only models (not synced with backend)
 */
export * from './message-bus';
