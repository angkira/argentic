# Messaging System Tests

This directory contains tests for the messaging system including unit tests and end-to-end (e2e) tests.

## Test Structure

- **Unit Tests**: Test functionality in isolation with mocks

  - `test_messager.py`: Tests for the main Messager class
  - `drivers/test_*_driver.py`: Tests for protocol-specific drivers
  - `test_messager_integration.py`: Tests for cross-protocol behavior with mocks

- **End-to-End Tests**: Test with real message brokers
  - `test_messager_e2e.py`: Verify actual message flow through brokers
  - `docker-compose.yml`: Container configuration for message brokers

## Running the Tests

### Unit Tests

Run unit tests without external dependencies:

```bash
# Run all messager tests
pytest tests/core/messager/

# Run specific test file
pytest tests/core/messager/test_messager.py

# Run tests for a specific driver
pytest tests/core/messager/drivers/test_mqtt_driver.py
```

### End-to-End Tests

These tests require Docker and docker compose. See `e2e_README.md` for detailed instructions.

```bash
# Run all e2e tests (requires Docker)
pytest tests/core/messager/test_messager_e2e.py -m e2e
```

## Test Coverage

The test suite provides comprehensive coverage of:

- Message encoding/decoding
- Connection management
- Publishing and subscribing
- Error handling
- Cross-protocol communication
- Actual message delivery through brokers

## Adding New Tests

When adding new tests:

1. For unit tests: Add to the appropriate file or create a new one if needed
2. For e2e tests: Add to `test_messager_e2e.py`
3. Run tests to verify they pass
