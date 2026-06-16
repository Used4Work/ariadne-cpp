# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |
| < 1.0   | No        |

## Reporting a Vulnerability

Email security issues to **yjxcwzh@outlook.com** with subject "[Ariadne Security]".

Please include:
- Description of the vulnerability
- Steps to reproduce
- Impact assessment
- Suggested fix (if any)

Response within 48 hours. Do not open public issues for security vulnerabilities.

## Security Measures

- API keys never logged (ILogger receives no credentials)
- Idempotency keys auto-generated per HTTP request
- CURLOPT_NOSIGNAL for POSIX thread safety
- CURLOPT_CONNECTTIMEOUT for fast failover
- Thread-safe ToolRegistry (shared_mutex)
- Safe JSON parse guards on all provider responses
- is_safe_id() path traversal prevention in Studio
- Studio bound to 127.0.0.1 (localhost only)
