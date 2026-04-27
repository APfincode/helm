# Threat Model — Hyper-Alpha-Arena V2

**Version:** 1.0  
**Date:** April 24, 2026  
**Classification:** Internal — Security Critical  

---

## 1. Trust Boundaries

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRUST BOUNDARIES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   [Untrusted Internet]                                                      │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐      │
│   │   Exchange API  │     │   LLM Provider  │     │ Telegram API    │      │
│   │  (Hyperliquid)  │     │  (OpenRouter)   │     │  (BotFather)    │      │
│   └────────┬────────┘     └────────┬────────┘     └────────┬────────┘      │
│            │                       │                       │               │
│   ═════════╧═══════════════════════╧═══════════════════════╧══════════     │
│                        NETWORK BOUNDARY (TLS)                               │
│   ═════════════════════════════════════════════════════════════════════     │
│            │                       │                       │               │
│   ┌────────▼────────┐     ┌───────▼─────────┐     ┌───────▼─────────┐      │
│   │   Bot Process   │◄────┤  LLM Response   │     │ Telegram Msg    │      │
│   │   (Python)      │     │  Validator      │     │ Handler         │      │
│   └────────┬────────┘     └─────────────────┘     └─────────────────┘      │
│            │                                                                │
│   ═════════╧═══════════════════════════════════════════════════════════     │
│                        PROCESS BOUNDARY                                      │
│   ═════════════════════════════════════════════════════════════════════     │
│            │                                                                │
│   ┌────────▼────────┐     ┌─────────────────┐     ┌─────────────────┐      │
│   │   SQLite DB     │     │   .env File     │     │   Audit Log     │      │
│   │   (trades)      │     │   (secrets)     │     │   (append-only) │      │
│   └─────────────────┘     └─────────────────┘     └─────────────────┘      │
│                                                                             │
│   ═════════════════════════════════════════════════════════════════════     │
│                        FILESYSTEM BOUNDARY                                   │
│   ═════════════════════════════════════════════════════════════════════     │
│                                                                             │
│   [Desktop OS — WSL/Windows]                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Threat Actors

| Actor | Motivation | Capability | Risk Level |
|-------|-----------|------------|------------|
| **Malware on Desktop** | Steal private keys, siphon funds | High (if bot has keys in memory) | 🔴 Critical |
| **Phishing Attacker** | Trick user into revealing credentials | Medium (social engineering) | 🟠 High |
| **Exchange Compromise** | Manipulate prices, freeze withdrawals | Low (exchange-level, not bot) | 🟡 Medium |
| **Supply Chain Attacker** | Inject malicious dependency | Medium (PyPI compromise) | 🟠 High |
| **LLM Prompt Injector** | Manipulate trading signals | Low (output validated) | 🟢 Low |
| **Insider (User Error)** | Misconfigure bot, leak keys | High (human error) | 🟠 High |
| **Network Eavesdropper** | Intercept API traffic | Very Low (TLS enforced) | 🟢 Low |

---

## 3. Attack Surfaces

### 3.1 Configuration Files
- **Risk:** `.env` file leaked via accidental commit, backup, or copy
- **Mitigation:** 
  - `.gitignore` blocks `.env` aggressively
  - `.env` permissions: `chmod 600`
  - Pre-commit hooks scan for secrets
  - Secrets manager refuses to read from anywhere else

### 3.2 SQLite Database
- **Risk:** Tampering with trade history, P&L manipulation
- **Mitigation:**
  - Audit log is append-only (DB trigger blocks UPDATE/DELETE)
  - Each row signed with HMAC
  - Database file permissions: `chmod 600`

### 3.3 Log Files
- **Risk:** Secrets accidentally logged (API keys, private keys)
- **Mitigation:**
  - Secrets manager masks all secrets in logs
  - `SecretStr` type from Pydantic never serializes raw values
  - Bandit scans for `password=`, `key=`, `secret=` in log calls

### 3.4 Network Traffic
- **Risk:** Man-in-the-middle on API calls
- **Mitigation:**
  - All API calls use TLS 1.3
  - Certificate pinning for exchange APIs
  - No HTTP (only HTTPS) allowed

### 3.5 Telegram Bot
- **Risk:** Unauthorized chat ID sending commands
- **Mitigation:**
  - Whitelist enforced on every command
  - Rate limiting per chat ID
  - Sensitive commands require HMAC token

### 3.6 LLM Output
- **Risk:** Prompt injection causing bad trades
- **Mitigation:**
  - Strict Pydantic validation on all LLM output
  - Prompt injection pattern detection
  - Invalid output defaults to NEUTRAL signal

### 3.7 Memory
- **Risk:** Private key extracted from process memory
- **Mitigation:**
  - Private key loaded once, used via SDK (never stored as string)
  - Process runs with least privilege
  - No core dumps enabled

---

## 4. Attack Scenarios

### Scenario 1: Malware Extracts Private Key from Memory
**Steps:**
1. Malware infects desktop
2. Scans process memory for hex strings matching private key format
3. Extracts key and drains wallet

**Defenses:**
- Bot runs in isolated process
- Private key accessed through SDK only (not held as Python string long-term)
- File permissions prevent other users from reading process memory
- Regular system updates and antivirus

### Scenario 2: Accidental `.env` Commit
**Steps:**
1. Developer copies `.env.example` to `.env`
2. Fills in real keys
3. Accidentally runs `git add .` without checking
4. Pushes to GitHub
5. Keys exposed in commit history

**Defenses:**
- `.gitignore` blocks `.env` at root and all subdirectories
- Pre-commit hook runs `detect-secrets` or `truffleHog`
- `.env.example` has placeholder values that fail validation
- GitHub secret scanning enabled (if using private repo)

### Scenario 3: Unauthorized Telegram Command
**Steps:**
1. Attacker discovers bot token or guesses bot username
2. Sends `/trade` command from unauthorized chat
3. Bot executes trade without verifying chat ID

**Defenses:**
- Every command checks `chat_id` against `TELEGRAM_WHITELIST`
- Whitelist loaded from `.env`, not hardcoded
- Unauthorized commands logged with chat ID and IP
- Rate limiting prevents brute-force of chat IDs

### Scenario 4: LLM Prompt Injection
**Steps:**
1. Attacker crafts market data with injection payload
2. LLM output contains malicious trading signal
3. Bot executes trade based on manipulated signal

**Defenses:**
- Input validator scans for injection patterns
- LLM output validated against strict Pydantic schema
- Signal defaults to NEUTRAL if validation fails
- Reasoning field sanitized (no control characters)

### Scenario 5: Database Tampering
**Steps:**
1. Attacker gains filesystem access
2. Modifies SQLite database to hide losses
3. Bot thinks it's profitable when it's not

**Defenses:**
- Audit log is append-only (DB trigger)
- Each row has HMAC signature
- Signature verification on every read
- Separate read-only user for analytics (if using PostgreSQL)

---

## 5. Risk Register

| ID | Risk | Likelihood | Impact | Status | Owner |
|----|------|-----------|--------|--------|-------|
| R1 | Private key leaked via `.env` commit | Medium | Critical | Mitigated | Dev |
| R2 | Malware extracts key from memory | Low | Critical | Mitigated | User |
| R3 | Unauthorized Telegram trade | Low | High | Mitigated | Dev |
| R4 | LLM prompt injection | Low | Medium | Mitigated | Dev |
| R5 | Database tampering | Very Low | High | Mitigated | Dev |
| R6 | Dependency supply chain attack | Low | High | Monitoring | Dev |
| R7 | Exchange API rate limit hit | Medium | Low | Mitigated | Dev |
| R8 | Circuit breaker bypassed | Very Low | Critical | Mitigated | Dev |

---

## 6. Security Controls Mapping

| Control | Threats Addressed | Implementation |
|---------|------------------|----------------|
| **HMAC Token Auth** | Unauthorized access | `security/auth.py` |
| **Telegram Whitelist** | Unauthorized commands | `security/auth.py` |
| **Secrets Manager** | Key leakage in logs | `security/secrets_manager.py` |
| **Input Validator** | Injection attacks | `security/input_validator.py` |
| **Audit Logger** | Tampering, accountability | `security/audit_logger.py` |
| **Rate Limiter** | Spam, brute force | `security/rate_limiter.py` |
| **Circuit Breakers** | Cascading losses | `risk/circuit_breaker.py` |
| **Pydantic Validation** | Invalid data, injection | `config/models.py` |
| **Pre-commit Hooks** | Secret leakage | `.pre-commit-config.yaml` |
| **Append-only Audit** | Database tampering | `security/audit_logger.py` |

---

## 7. Incident Response

### 7.1 Key Compromise
1. **Immediate:** Run `/halt` on Telegram bot (stops all trading)
2. **Within 5 min:** Transfer funds to new wallet
3. **Within 1 hour:** Revoke old API keys, generate new ones
4. **Within 24 hours:** Audit all trades since compromise
5. **Document:** Record incident in `docs/incidents/`

### 7.2 Unauthorized Trade Detected
1. **Immediate:** Run `/halt`
2. **Investigate:** Check audit log for entry point
3. **Assess:** Determine if position should be closed
4. **Patch:** Fix vulnerability that allowed unauthorized access
5. **Document:** Record incident

### 7.3 Bot Malfunction
1. **Immediate:** Run `/halt`
2. **Check:** Review last 50 audit log entries
3. **Diagnose:** Check circuit breaker status, config changes
4. **Rollback:** Revert to last known good config
5. **Test:** Run paper trading before re-enabling live

---

## 8. Compliance & Assumptions

### Assumptions
- Desktop OS is kept up-to-date with security patches
- User does not run bot as Administrator/root
- Physical access to device is controlled
- Network is not compromised at the router level

### Compliance
- This bot is for personal use only
- No KYC/AML requirements for self-custodial trading
- User is responsible for tax reporting
- Bot does not handle customer funds

---

*Last updated: April 24, 2026*  
*Next review: Before Phase 7 (live trading)*
