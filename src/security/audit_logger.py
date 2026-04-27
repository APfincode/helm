"""
Audit Logger — Immutable, append-only, signed audit trail.

Every sensitive action is logged here:
- Trades (entry, exit, P&L)
- Config changes (before/after state)
- Authentication events (success/failure)
- System events (startup, shutdown, circuit breaker triggers)
- Errors and exceptions

Design:
- Append-only: UPDATE and DELETE are blocked at DB trigger level
- Tamper-evident: Each row has HMAC signature
- Structured: JSON fields for flexible event data

Usage:
    from src.security.audit_logger import AuditLogger
    
    async with AuditLogger() as logger:
        await logger.log_trade(
            actor="telegram:123456789",
            action="OPEN_LONG",
            details={"asset": "BTC", "size": 0.1, "price": 65000},
        )
"""

import hmac
import hashlib
import json
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass, asdict

import aiosqlite

from src.security.secrets_manager import SecretsManager, get_secrets_manager


class AuditError(Exception):
    """Audit logging error."""
    pass


class EventType(str, Enum):
    """Types of audit events."""
    TRADE = "trade"
    CONFIG_CHANGE = "config_change"
    AUTH = "auth"
    AUTH_FAILURE = "auth_failure"
    ERROR = "error"
    SYSTEM = "system"
    CIRCUIT_BREAKER = "circuit_breaker"
    LLM_SIGNAL = "llm_signal"
    RISK_CHECK = "risk_check"


class Severity(str, Enum):
    """Event severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class AuditEntry:
    """Single audit log entry."""
    id: Optional[int] = None
    timestamp: str = ""
    event_type: str = ""
    severity: str = ""
    actor: str = ""
    action: str = ""
    details: Optional[dict[str, Any]] = None
    before_state: Optional[dict[str, Any]] = None
    after_state: Optional[dict[str, Any]] = None
    signature: str = ""


class AuditLogger:
    """
    Async audit logger with tamper-evident signatures.
    
    SQLite database with triggers enforcing append-only behavior.
    """

    def __init__(
        self,
        db_path: str = "data/audit.db",
        signing_key: Optional[str] = None,
    ) -> None:
        self._db_path = db_path
        self._signing_key = signing_key
        self._db: Optional[aiosqlite.Connection] = None
        self._initialized = False

    async def __aenter__(self) -> "AuditLogger":
        await self._init_db()
        return self

    async def __aexit__(self, *args) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _init_db(self) -> None:
        """Initialize database with schema and append-only triggers."""
        if self._initialized:
            return

        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)

        self._db = await aiosqlite.connect(self._db_path)
        
        # Create table
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                before_state TEXT,
                after_state TEXT,
                signature TEXT NOT NULL
            )
        """)
        
        # Create indexes
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
            ON audit_log(timestamp)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_event_type 
            ON audit_log(event_type)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_actor 
            ON audit_log(actor)
        """)
        
        # Append-only trigger: block UPDATE
        await self._db.execute("""
            CREATE TRIGGER IF NOT EXISTS audit_prevent_update
            BEFORE UPDATE ON audit_log
            BEGIN
                SELECT RAISE(FAIL, 'Audit log is append-only: updates not allowed');
            END
        """)
        
        # Append-only trigger: block DELETE
        await self._db.execute("""
            CREATE TRIGGER IF NOT EXISTS audit_prevent_delete
            BEFORE DELETE ON audit_log
            BEGIN
                SELECT RAISE(FAIL, 'Audit log is append-only: deletes not allowed');
            END
        """)
        
        await self._db.commit()
        self._initialized = True

    def _get_signing_key(self) -> str:
        """Get or load signing key."""
        if self._signing_key is None:
            secrets_mgr = get_secrets_manager()
            self._signing_key = secrets_mgr.get("HMAC_SECRET_KEY", required=True)
        return self._signing_key

    def _sign_entry(self, entry: AuditEntry) -> str:
        """Create HMAC signature for an audit entry."""
        # Build payload from immutable fields
        payload = json.dumps({
            "timestamp": entry.timestamp,
            "event_type": entry.event_type,
            "severity": entry.severity,
            "actor": entry.actor,
            "action": entry.action,
            "details": entry.details,
            "before_state": entry.before_state,
            "after_state": entry.after_state,
        }, sort_keys=True, separators=(",", ":"))
        
        return hmac.new(
            self._get_signing_key().encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def log(
        self,
        event_type: EventType,
        action: str,
        actor: str = "system",
        severity: Severity = Severity.INFO,
        details: Optional[dict[str, Any]] = None,
        before_state: Optional[dict[str, Any]] = None,
        after_state: Optional[dict[str, Any]] = None,
    ) -> int:
        """
        Log an audit event.
        
        Args:
            event_type: Category of event
            action: Human-readable action description
            actor: Who/what performed the action
            severity: Event severity
            details: Additional event data
            before_state: State before change (for config changes)
            after_state: State after change (for config changes)
            
        Returns:
            Row ID of inserted log entry
        """
        if self._db is None:
            raise AuditError("Database not initialized. Use 'async with' context.")

        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type.value,
            severity=severity.value,
            actor=actor,
            action=action,
            details=details,
            before_state=before_state,
            after_state=after_state,
        )
        
        entry.signature = self._sign_entry(entry)

        cursor = await self._db.execute(
            """
            INSERT INTO audit_log 
            (timestamp, event_type, severity, actor, action, details, before_state, after_state, signature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.timestamp,
                entry.event_type,
                entry.severity,
                entry.actor,
                entry.action,
                json.dumps(entry.details) if entry.details else None,
                json.dumps(entry.before_state) if entry.before_state else None,
                json.dumps(entry.after_state) if entry.after_state else None,
                entry.signature,
            )
        )
        await self._db.commit()
        return cursor.lastrowid

    async def verify_entry(self, entry_id: int) -> bool:
        """
        Verify an audit entry's signature.
        
        Args:
            entry_id: Row ID to verify
            
        Returns:
            True if signature is valid, False if tampered
        """
        if self._db is None:
            raise AuditError("Database not initialized.")

        async with self._db.execute(
            "SELECT * FROM audit_log WHERE id = ?", (entry_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return False

        entry = AuditEntry(
            id=row[0],
            timestamp=row[1],
            event_type=row[2],
            severity=row[3],
            actor=row[4],
            action=row[5],
            details=json.loads(row[6]) if row[6] else None,
            before_state=json.loads(row[7]) if row[7] else None,
            after_state=json.loads(row[8]) if row[8] else None,
            signature=row[9],
        )

        expected_sig = self._sign_entry(entry)
        return hmac.compare_digest(entry.signature, expected_sig)

    async def get_recent(
        self,
        event_type: Optional[EventType] = None,
        actor: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Retrieve recent audit entries with optional filtering."""
        if self._db is None:
            raise AuditError("Database not initialized.")

        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        if actor:
            query += " AND actor = ?"
            params.append(actor)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        entries = []
        async with self._db.execute(query, params) as cursor:
            async for row in cursor:
                entries.append(AuditEntry(
                    id=row[0],
                    timestamp=row[1],
                    event_type=row[2],
                    severity=row[3],
                    actor=row[4],
                    action=row[5],
                    details=json.loads(row[6]) if row[6] else None,
                    before_state=json.loads(row[7]) if row[7] else None,
                    after_state=json.loads(row[8]) if row[8] else None,
                    signature=row[9],
                ))

        return entries

    # Convenience methods
    async def log_trade(
        self,
        action: str,
        actor: str = "system",
        details: Optional[dict[str, Any]] = None,
        severity: Severity = Severity.INFO,
    ) -> int:
        """Log a trade event."""
        return await self.log(
            event_type=EventType.TRADE,
            action=action,
            actor=actor,
            severity=severity,
            details=details,
        )

    async def log_config_change(
        self,
        action: str,
        actor: str,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        severity: Severity = Severity.INFO,
    ) -> int:
        """Log a configuration change with before/after state."""
        return await self.log(
            event_type=EventType.CONFIG_CHANGE,
            action=action,
            actor=actor,
            severity=severity,
            before_state=before_state,
            after_state=after_state,
        )

    async def log_auth(
        self,
        action: str,
        actor: str,
        success: bool,
        details: Optional[dict[str, Any]] = None,
    ) -> int:
        """Log an authentication event."""
        event_type = EventType.AUTH if success else EventType.AUTH_FAILURE
        severity = Severity.INFO if success else Severity.WARNING
        return await self.log(
            event_type=event_type,
            action=action,
            actor=actor,
            severity=severity,
            details=details,
        )

    async def log_error(
        self,
        action: str,
        error_message: str,
        actor: str = "system",
        severity: Severity = Severity.ERROR,
    ) -> int:
        """Log an error event."""
        return await self.log(
            event_type=EventType.ERROR,
            action=action,
            actor=actor,
            severity=severity,
            details={"error": error_message},
        )

    async def log_circuit_breaker(
        self,
        breaker_name: str,
        triggered: bool,
        actor: str = "system",
        details: Optional[dict[str, Any]] = None,
    ) -> int:
        """Log a circuit breaker event."""
        action = f"CIRCUIT_BREAKER_{'TRIGGERED' if triggered else 'RESET'}:{breaker_name}"
        return await self.log(
            event_type=EventType.CIRCUIT_BREAKER,
            action=action,
            actor=actor,
            severity=Severity.WARNING if triggered else Severity.INFO,
            details=details,
        )
