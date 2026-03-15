# Security Design Document

This document describes the security architecture of mcp-trove-crunchtools.

## 1. Threat Model

### 1.1 Assets to Protect

| Asset | Sensitivity | Impact if Compromised |
|-------|-------------|----------------------|
| SQLite Database | Medium | Index metadata, embeddings exposed |
| Indexed File Content | High | Text chunks from local files exposed |
| Local File Paths | Medium | Directory structure revealed |

### 1.2 Threat Actors

| Actor | Capability | Motivation |
|-------|------------|------------|
| Malicious AI Agent | Can craft tool inputs | Data exfiltration, path traversal |
| Local Attacker | Access to filesystem | Database tampering, content theft |

### 1.3 Attack Vectors

| Vector | Description | Mitigation |
|--------|-------------|------------|
| **Path Traversal** | Manipulated file paths in index/search | Input validation, path canonicalization |
| **SQL Injection** | Crafted tool inputs | Parameterized queries only |
| **Denial of Service** | Index extremely large files | File size limits, batch limiting |
| **Content Injection** | Crafted file names or content | Content stored as-is, safe queries |

## 2. Security Architecture

### 2.1 Defense in Depth Layers

```
+---------------------------------------------------------+
| Layer 1: Credential Protection                           |
| - N/A — local files only, no external API credentials   |
| - SecretStr pattern available if credentials added later |
+---------------------------------------------------------+
| Layer 2: Input Validation                                |
| - Pydantic models for all tool inputs                   |
| - Reject unexpected fields (extra="forbid")             |
| - Field length limits, path validation                   |
+---------------------------------------------------------+
| Layer 3: File System Hardening                           |
| - Read-only file access (no writes to indexed files)    |
| - Path canonicalization to prevent traversal             |
| - File size limits to prevent memory exhaustion          |
| - Exclude patterns for binary/large files                |
+---------------------------------------------------------+
| Layer 4: Runtime Protection                              |
| - No shell execution (subprocess)                        |
| - No dynamic code evaluation (eval/exec)                |
| - asyncio.Semaphore for resource limiting                |
| - Batch size limits on indexing operations               |
+---------------------------------------------------------+
| Layer 5: Supply Chain Security                           |
| - Automated CVE scanning via GitHub Actions             |
| - Container built on Hummingbird for minimal CVEs       |
| - Weekly dependency audits                               |
+---------------------------------------------------------+
```

### 2.2 No Credentials

This server has no API tokens or credentials. It reads local files and stores embeddings in a local SQLite database. The primary security concerns are input validation and file access control.

### 2.3 Input Validation Rules

All inputs are validated:

- **File paths**: Canonicalized, must exist on filesystem
- **Search queries**: String length limited
- **Limits/offsets**: Bounded integers
- **Extra Fields**: Rejected (Pydantic extra="forbid")

## 3. Supply Chain Security

### 3.1 Automated CVE Scanning

This project uses GitHub Actions to automatically scan for CVEs:

1. **Weekly Scheduled Scans**: Every Monday at 9 AM UTC
2. **PR Checks**: Every pull request is scanned before merge
3. **Dependabot**: Enabled for automatic security updates

### 3.2 Container Security

The container image is built on **[Hummingbird Python](https://quay.io/repository/hummingbird/python)** from [Project Hummingbird](https://github.com/hummingbird-project):

| Advantage | Description |
|-----------|-------------|
| **Minimal CVE Count** | Dramatically reduced attack surface |
| **Rapid Security Updates** | Security patches applied promptly |
| **Python Optimized** | Pre-configured with uv package manager |
| **Non-Root Default** | Runs as non-root user |
| **Production Ready** | Proper signal handling, minimal footprint |

## 4. Security Checklist

Before each release:

- [ ] All inputs validated through Pydantic models
- [ ] File paths canonicalized before access
- [ ] No shell execution
- [ ] No eval/exec
- [ ] Error messages don't leak internals
- [ ] Dependencies scanned for CVEs
- [ ] Container rebuilt with latest Hummingbird base

## 5. Reporting Security Issues

Report security vulnerabilities using [GitHub's private security advisory](https://github.com/crunchtools/mcp-trove/security/advisories/new). This creates a private channel visible only to maintainers.

Do NOT open public issues for security vulnerabilities.
