# JOI Test Suite

This directory contains all tests for the JOI English Learning Assistant system.

## Test Files

### 🧪 System Tests
- **`comprehensive_system_test.py`** - Complete system validation (14 components)
- **`test_assessment_system.py`** - Assessment engine tests
- **`test_curriculum.py`** - Curriculum management tests
- **`test_connections.py`** - Database and API connection tests
- **`test_whatsapp_api.py`** - WhatsApp integration tests
- **`test_postgres_checkpointer.py`** - PostgreSQL checkpointer tests

## Running Tests

### Run All Tests
```bash
# From project root
python test/comprehensive_system_test.py
```

### Run Individual Tests
```bash
python test/test_assessment_system.py
python test/test_curriculum.py
python test/test_connections.py
```

### Requirements
- App must be running on `http://localhost:8000`
- Valid database connection
- Environment variables configured (.env file)

## Test Coverage

The comprehensive test covers:
- ✅ Database Connectivity
- ✅ Database Schema
- ✅ API Health Check
- ✅ Basic API Endpoints
- ✅ User Management
- ✅ Session Management
- ✅ Message System
- ✅ Memory System
- ✅ Curriculum System
- ✅ Assessment System
- ✅ Debug Endpoints
- ✅ Integration Tests
- ✅ Error Handling
- ✅ Performance Tests

## Success Criteria

- **100% Success Rate** = All systems operational, ready for production
- **80-99% Success Rate** = Mostly operational, minor issues to address
- **<80% Success Rate** = Significant issues, review required

## Notes

- Tests use real user data from the system
- Some tests modify database state (assessments, progress)
- All import paths are configured for the test/ directory location 