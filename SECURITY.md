# Security Policy

## Supported Versions

Security fixes are handled on the `main` branch until the project publishes a
formal support matrix.

## Reporting a Vulnerability

Please report security issues privately instead of opening a public issue.

Email: security@axolotl.ai

Include:

- affected version or commit
- impact and attack scenario
- reproduction steps or proof of concept
- any relevant logs, configuration, or environment details

We will acknowledge reports as soon as practical and coordinate disclosure once
a fix or mitigation is available.

## Deployment Notes

- Set a strong `HATCHERY_ADMIN_API_KEY` for any exposed gateway.
- Run the gateway behind TLS on untrusted networks.
- Treat object-store contents, checkpoints, and Hugging Face caches as
  sensitive model data.
- Use model allow-lists and network egress controls for shared deployments.
