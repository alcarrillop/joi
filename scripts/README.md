# JOI Scripts

This directory contains setup scripts and utilities for the JOI English Learning Assistant system.

## Database Setup Scripts

### 📊 Table Creation Scripts
- **`create_assessment_tables.py`** - Creates assessment system tables, views, and triggers
- **`create_curriculum_tables.py`** - Creates curriculum system tables and views  
- **`create_missing_tables.py`** - Creates missing tables (learning_modules, competencies)

### 📄 SQL Schema Files
- **`assessment_tables.sql`** - Assessment system database schema
- **`curriculum_tables.sql`** - Curriculum system database schema
- **`init_db.sql`** - Initial database setup

## Utility Scripts

### 🔧 Debug Tools
- **`debug_tools.py`** - Command-line debugging and monitoring tools

## Usage

### Database Setup
```bash
# Create assessment tables
python scripts/create_assessment_tables.py

# Create curriculum tables  
python scripts/create_curriculum_tables.py

# Create missing tables (if needed)
python scripts/create_missing_tables.py
```

### Debug Tools
```bash
# Interactive menu
python scripts/debug_tools.py

# Direct commands
python scripts/debug_tools.py stats
python scripts/debug_tools.py health
python scripts/debug_tools.py users
python scripts/debug_tools.py analyze
```

## Requirements

- Valid database connection (DATABASE_URL in .env)
- Python environment with required packages
- For debug tools: running app on localhost:8000

## Script Functions

### Table Creation Scripts
- ✅ Create database tables and indexes
- ✅ Create views for data analysis
- ✅ Set up triggers for automatic updates
- ✅ Insert sample data (where applicable)
- ✅ Verify table creation and data integrity

### Debug Tools
- 📊 System statistics and health checks
- 👥 User management and analysis
- 💬 Message and session inspection
- 🧠 Memory system testing
- 🎓 Curriculum progress monitoring
- 📈 Performance monitoring

## Notes

- All scripts include proper error handling and verification
- Scripts can be run multiple times safely (CREATE IF NOT EXISTS)
- Import paths are configured for the scripts/ directory location 