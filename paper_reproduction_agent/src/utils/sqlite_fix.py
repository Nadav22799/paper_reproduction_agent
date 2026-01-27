"""SQLite3 compatibility fix for ChromaDB.

Overrides the default sqlite3 module with pysqlite3 to satisfy ChromaDB's
version requirements (>=3.35.0). Works for both Windows (via pysqlite3-binary usually)
and Linux/WSL (via pysqlite3 if built correctly).
"""
import sys
import logging

logger = logging.getLogger(__name__)

def apply_sqlite_fix():
    """Apply the SQLite3 fix by swapping the module for pysqlite3."""
    try:
        # Check if we're on Windows (primary target for this issue)
        # or if explicitly needed
        import platform
        
        # Only apply if standard sqlite3 is too old
        import sqlite3
        sqlite_version = sqlite3.sqlite_version
        
        # Parse version
        major, minor, patch = map(int, sqlite_version.split('.')[:3])
        
        # ChromaDB requires >= 3.35.0
        if (major < 3) or (major == 3 and minor < 35):
            logger.warning(f"Detected outdated sqlite3 version: {sqlite_version}. Attempting override...")
            
            try:
                __import__('pysqlite3')
                sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
                
                # Verify the fix
                import sqlite3
                logger.info(f"Successfully overrode sqlite3. New version: {sqlite3.sqlite_version}")
            except ImportError:
                logger.error("Failed to import pysqlite3. Please run: pip install pysqlite3-binary")
        else:
            logger.debug(f"SQLite3 version {sqlite_version} is compatible.")
            
    except Exception as e:
        logger.error(f"Failed to apply SQLite3 fix: {e}")
