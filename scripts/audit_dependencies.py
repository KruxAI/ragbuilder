"""Fail on new dependency advisories; document narrowly scoped unresolved risks."""
import json
import subprocess
import sys

# No patched upstream versions at the time of the security patch. See SECURITY.md.
EXCEPTIONS = {
    ("ragas", "GHSA-95ww-475f-pr4f"),
    ("diskcache", "GHSA-w8v5-vhqr-4h9v"),
    ("chromadb", "GHSA-f4j7-r4q5-qw2c"),
    ("chromadb", "GHSA-36p7-vc44-83pf"),
    ("chromadb", "GHSA-xph7-9rjv-w5fr"),
    ("chromadb", "GHSA-2wm9-hf6c-p5cr"),
}

result = subprocess.run([sys.executable, "-m", "pip_audit", "--format=json", "--progress-spinner=off"], capture_output=True, text=True)
if result.returncode not in {0, 1}:
    sys.exit(result.stderr or "Dependency audit failed")
try:
    report = json.loads(result.stdout)
except json.JSONDecodeError:
    sys.exit(result.stderr or "Dependency audit returned no report")
unexpected = []
for dependency in report["dependencies"]:
    for finding in dependency.get("vulns", []):
        identifiers = {finding["id"], *finding.get("aliases", [])}
        known = any((dependency["name"], identifier) in EXCEPTIONS for identifier in identifiers)
        # Once a fix exists, the old exception must no longer suppress the finding.
        accepted = known and not finding.get("fix_versions")
        print(f"{'DOCUMENTED' if accepted else 'ACTION REQUIRED'}: {dependency['name']} {finding['id']}")
        if not accepted:
            unexpected.append(finding)
sys.exit(bool(unexpected))
