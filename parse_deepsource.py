import json
from collections import defaultdict

with open('DS_AUDIT2.md', 'r') as f:
    content = f.read()

data = json.loads(content)
occurences = data['occurences']
summary = data['summary']

files = defaultdict(lambda: defaultdict(list))

for occ in occurences:
    path = occ['location']['path']
    issue_code = occ['issue_code']
    line = occ['location']['position']['begin']['line']
    title = occ['issue_title']
    files[path][issue_code].append({'line': line, 'title': title})

def get_severity(issue_code):
    if issue_code.startswith(('PYL-E', 'FLK-E', 'PY-E')):
        return 'Critical'
    elif issue_code.startswith(('PYL-W', 'FLK-W', 'PY-W')):
        return 'Major'
    else:
        return 'Minor'

report = f"# DEEPSOURCE AUDIT REPORT (Branch-wide Total Issues: {summary['total_occurences']} (CLI-verified))\n\n"
report += f"This report catalogs ALL issues identified through CLI analysis. Total occurrences: {summary['total_occurences']}, unique issues: {summary['unique_issues']}.\n\n---\n\n"

total_critical = 0
total_major = 0
total_minor = 0

for path in sorted(files.keys()):
    report += f"## Issues for {path}\n\n"
    issues_by_severity = defaultdict(list)
    for issue_code, occs in files[path].items():
        severity = get_severity(issue_code)
        lines = sorted(set(occ['line'] for occ in occs))
        count = len(occs)
        title = occs[0]['title']
        issues_by_severity[severity].append({
            'rule': issue_code,
            'description': title,
            'count': count,
            'lines': lines
        })

    for severity in ['Critical', 'Major', 'Minor']:
        issues = issues_by_severity[severity]
        if issues:
            report += f"### {severity} Issues ({len(issues)} total)\n"
            for issue in issues:
                report += f"- **Rule: {issue['rule']}** ({issue['description']}) - Total count: {issue['count']} instances.\n"
                report += f"  - Affected lines: {', '.join(map(str, issue['lines']))}\n"
            report += "\n"
        else:
            report += f"### {severity} Issues (0 total)\n- None identified.\n\n"

    total_critical += len(issues_by_severity['Critical'])
    total_major += len(issues_by_severity['Major'])
    total_minor += len(issues_by_severity['Minor'])

    report += "---\n\n"

report += f"**Audit Summary:** Total issues: {summary['total_occurences']} (Critical: {total_critical}, Major: {total_major}, Minor: {total_minor}).\n"

# Find C0301 if present
c0301_found = False
for path, issues in files.items():
    for issue_code in issues:
        if 'C0301' in issue_code:
            occs = issues[issue_code]
            lines = sorted(set(occ['line'] for occ in occs))
            report += f"\nC0301 example: {len(occs)} instances at lines {', '.join(map(str, lines))} in {path}\n"
            c0301_found = True
            break
    if c0301_found:
        break

if not c0301_found:
    report += "\nC0301 not found in the data.\n"

with open('DEEPSOURCE_AUDIT.md', 'w') as f:
    f.write(report)

print("Report updated successfully.")