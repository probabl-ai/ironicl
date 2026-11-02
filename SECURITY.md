# Security Policy

Ironicl reads PyTorch checkpoints, Ironicl model and context artifacts, and
request bundles. Treat every one of these files as untrusted input. A file that
makes Ironicl crash, hang, allocate unbounded memory, read out of bounds, or
execute code is a security issue.

## Supported Versions

Ironicl has no releases yet. Only the latest commit on `main` receives security
fixes.

## Reporting a Vulnerability

Do not open a public issue for a vulnerability.

Report it privately through
[GitHub private vulnerability reporting](https://github.com/probabl-ai/ironicl/security/advisories/new).
Include:

- the affected command or API;
- the Ironicl commit;
- the output of `ironicl platform`;
- a minimal file or steps that reproduce the problem.

The maintainers will confirm receipt, investigate, and coordinate a fix and
disclosure with you.

## Scope

In scope:

- the checkpoint importer and its pickle and ZIP handling;
- the model, static-context, input, and query artifact parsers;
- memory safety in native kernels;
- the Python preprocessing package.

Out of scope:

- prediction accuracy or numerical differences from TabICL;
- vulnerabilities in TabICL, PyTorch, or the released checkpoints themselves;
- denial of service from inputs that are within documented limits.

See [docs/security.md](docs/security.md) for the parser and data-handling
design.
