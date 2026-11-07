# Security Policy

The RHEN team takes the security and integrity of agentic systems seriously. As RHEN governs autonomous agent decision-making, cryptographic execution gating, and audit logging, we treat security vulnerabilities with high urgency.

---

## Supported Versions

Only the latest stable release receives active security patches.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a potential vulnerability in:
1. **Cryptographic Action Card signing & HMAC-SHA256 verification** (e.g. signature bypasses or canonical JSON serialization flaws).
2. **Dual-custody co-signing & BFT consensus integrity**.
3. **Out-of-band audit ledger sealing and tamper resistance**.
4. **Delimiter token sanitization & prompt injection escapes** (e.g. control tokens escaping `<|` / `|>`).
5. **Conformal prediction coverage calculation or geometric abstention gates**.

Please report it privately by emailing the maintainers or opening a private [GitHub Security Advisory](https://github.com/Kivitas/RHEN/security/advisories/new).

### What to Include in Your Report:
- A clear description of the vulnerability.
- A minimal Python reproduction script.
- The affected component (`agent_tools.py`, `rhen_verify.py`, `rhen_packer.py`, etc.).
- Potential impact and suggested mitigations if known.

We will acknowledge receipt of your report within 48 hours and work with you on a responsible disclosure timeline.
