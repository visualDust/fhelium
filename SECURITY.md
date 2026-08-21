# FHElium security policy

This policy defines the private vulnerability-reporting process for FHElium.
The [security scope documentation](https://fhelium.550w.host/tutorial/support-and-security)
describes the properties FHElium validates and the security responsibilities
owned by applications.

## Report a vulnerability privately

Submit suspected vulnerabilities through
[GitHub Private Vulnerability Reporting](https://github.com/VisualDust/fhelium/security/advisories/new).
Do not open a public issue or discussion for an undisclosed vulnerability.

Include the following information when available:

- affected FHElium version or commit and installation source;
- operating system, architecture, Python, Torch, and CUDA environment when
  relevant;
- affected API, native operator, file format, build path, or release artifact;
- vulnerability description and potential security impact;
- minimal reproduction using synthetic data and throwaway keys;
- conditions required to trigger the problem;
- known mitigations or related public information;
- whether the report or any part of it has been disclosed elsewhere.

Do not submit credentials, production secret keys, proprietary plaintexts, or
other private data. GitHub's private advisory discussion can be used for
follow-up evidence and coordination.

## Security-relevant reports

Private reports are appropriate for suspected weaknesses such as:

- disclosure or unintended reuse of secret key material;
- incorrect parameter or state validation that weakens a documented security
  property;
- memory-safety defects in FHElium native code;
- unsafe deserialization or artifact handling with a security impact;
- cross-context, cross-process, or cross-device data exposure caused by
  FHElium;
- compromise of FHElium's build, package, or release-artifact integrity.

Ordinary functional defects, numerical discrepancies without a security
impact, installation problems, performance reports, and feature requests
belong in GitHub Issues or Discussions.

## Coordination and disclosure

Maintainers use the private advisory to assess impact, request additional
evidence, develop and validate a fix, and coordinate disclosure. Please keep
the report private until a remediation and disclosure plan has been agreed.
The published advisory may credit reporters who wish to be identified.
