#!/usr/bin/env python3
"""
Security Audit Script for SAMO-DL Demo Website

This script performs comprehensive security analysis of the demo website including:
- XSS vulnerability detection
- Input sanitization validation
- API communication security
- Content Security Policy validation
- Dependency security scanning

Usage:
    python scripts/security_audit_demo.py [--verbose] [--output-format json|html]
"""
import sys
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class DemoSecurityAuditor:
    """Security auditor for demo website"""

    def __init__(self, verbose=False, output_format='json'):
        self.verbose = verbose
        self.output_format = output_format
        self.website_path = project_root / "website"
        self.audit_results = {
            'timestamp': datetime.now().isoformat(),
            'vulnerabilities': [],
            'warnings': [],
            'recommendations': [],
            'security_score': 0,
            'summary': {}
        }

    def run_security_audit(self):
        """Run comprehensive security audit"""
        print("🔒 Starting SAMO-DL Demo Website Security Audit")
        print("=" * 60)

        # Check if website directory exists
        if not self.website_path.exists():
            print(f"❌ Website directory not found: {self.website_path}")
            return

        # Run security checks
        self.check_xss_vulnerabilities()
        self.check_input_sanitization()
        self.check_api_security()
        self.check_csp_headers()
        self.check_dependency_security()
        self.check_file_permissions()
        self.check_sensitive_data_exposure()
        self.check_authentication_security()

        # Calculate security score
        self.calculate_security_score()

        # Generate report
        self.generate_security_report()

    def check_xss_vulnerabilities(self):
        """Check for XSS vulnerabilities in JavaScript files"""
        print("\n🔍 Checking for XSS vulnerabilities...")

        js_files = list(self.website_path.glob("**/*.js"))
        xss_patterns = [
            r'innerHTML\s*=',
            r'outerHTML\s*=',
            r'document\.write\(',
            r'eval\(',
            r'setTimeout\s*\(\s*["\'].*["\']',
            r'setInterval\s*\(\s*["\'].*["\']',
            r'Function\s*\(',
            r'new\s+Function\s*\('
        ]

        for js_file in js_files:
            try:
                with open(js_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                    for i, line in enumerate(lines, 1):
                        for pattern in xss_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                vulnerability = {
                                    'type': 'XSS',
                                    'severity': 'HIGH',
                                    'file': str(js_file.relative_to(project_root)),
                                    'line': i,
                                    'code': line.strip(),
                                    'description': f'Potential XSS vulnerability: {pattern}',
                                    'recommendation': 'Use textContent instead of innerHTML, avoid eval(), and sanitize user input'
                                }
                                self.audit_results['vulnerabilities'].append(vulnerability)

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error reading {js_file}: {e}")

    def check_input_sanitization(self):
        """Check input sanitization in form handling"""
        print("🔍 Checking input sanitization...")

        js_files = list(self.website_path.glob("**/*.js"))
        input_patterns = [
            r'\.value\s*=',
            r'\.textContent\s*=',
            r'\.innerHTML\s*=',
            r'JSON\.parse\(',
            r'JSON\.stringify\('
        ]

        for js_file in js_files:
            try:
                with open(js_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                    for i, line in enumerate(lines, 1):
                        for pattern in input_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                # Check if there's sanitization nearby
                                sanitization_found = any(
                                    'sanitize' in lines[j].lower() or 
                                    'escape' in lines[j].lower() or
                                    'validate' in lines[j].lower()
                                    for j in range(max(0, i-3), min(len(lines), i+3))
                                )

                                if not sanitization_found:
                                    warning = {
                                        'type': 'INPUT_SANITIZATION',
                                        'severity': 'MEDIUM',
                                        'file': str(js_file.relative_to(project_root)),
                                        'line': i,
                                        'code': line.strip(),
                                        'description': 'Input handling without apparent sanitization',
                                        'recommendation': 'Implement input validation and sanitization'
                                    }
                                    self.audit_results['warnings'].append(warning)

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error reading {js_file}: {e}")

    def check_api_security(self):
        """Check API communication security"""
        print("🔍 Checking API security...")

        js_files = list(self.website_path.glob("**/*.js"))
        api_patterns = [
            r'fetch\s*\(',
            r'XMLHttpRequest',
            r'axios\s*\.',
            r'\.post\s*\(',
            r'\.get\s*\('
        ]

        for js_file in js_files:
            try:
                with open(js_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                    for i, line in enumerate(lines, 1):
                        for pattern in api_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                # Check for HTTPS usage
                                if 'http://' in line and 'https://' not in line:
                                    vulnerability = {
                                        'type': 'INSECURE_HTTP',
                                        'severity': 'HIGH',
                                        'file': str(js_file.relative_to(project_root)),
                                        'line': i,
                                        'code': line.strip(),
                                        'description': 'HTTP request detected (should use HTTPS)',
                                        'recommendation': 'Use HTTPS for all API communications'
                                    }
                                    self.audit_results['vulnerabilities'].append(vulnerability)

                                # Check for API key exposure
                                if (
                                    'api' in line.lower()
                                    and 'key' in line.lower()
                                    and 'process.env' not in line
                                    and 'config' not in line.lower()
                                ):
                                    warning = {
                                        'type': 'API_KEY_EXPOSURE',
                                        'severity': 'HIGH',
                                        'file': str(js_file.relative_to(project_root)),
                                        'line': i,
                                        'code': line.strip(),
                                        'description': 'Potential API key exposure in client-side code',
                                        'recommendation': 'Use environment variables or secure configuration'
                                    }
                                    self.audit_results['vulnerabilities'].append(warning)

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error reading {js_file}: {e}")

    def check_csp_headers(self):
        """Check Content Security Policy implementation"""
        print("🔍 Checking Content Security Policy...")

        html_files = list(self.website_path.glob("**/*.html"))
        csp_found = False

        for html_file in html_files:
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                    if 'Content-Security-Policy' in content or 'content-security-policy' in content:
                        csp_found = True
                        break

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error reading {html_file}: {e}")

        if not csp_found:
            warning = {
                'type': 'MISSING_CSP',
                'severity': 'MEDIUM',
                'file': 'All HTML files',
                'line': 0,
                'code': 'N/A',
                'description': 'Content Security Policy not implemented',
                'recommendation': 'Implement CSP headers to prevent XSS attacks'
            }
            self.audit_results['warnings'].append(warning)

    def check_dependency_security(self):
        """Check for security vulnerabilities in dependencies"""
        print("🔍 Checking dependency security...")

        # Check package.json if it exists
        package_json = self.website_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json, 'r') as f:
                    package_data = json.load(f)

                # Check for known vulnerable packages
                vulnerable_packages = [
                    'jquery@1.x',
                    'lodash@4.17.0',
                    'moment@2.19.0'
                ]

                dependencies = package_data.get('dependencies', {})
                for package_name, version in dependencies.items():
                    if any(vuln_pkg.split('@')[0] in package_name for vuln_pkg in vulnerable_packages):
                        warning = {
                            'type': 'VULNERABLE_DEPENDENCY',
                            'severity': 'HIGH',
                            'file': 'package.json',
                            'line': 0,
                            'code': f'{package_name}: {version}',
                            'description': f'Potentially vulnerable dependency: {package_name}',
                            'recommendation': 'Update to latest secure version'
                        }
                        self.audit_results['warnings'].append(warning)

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error reading package.json: {e}")

        # Check CDN dependencies in HTML files
        html_files = list(self.website_path.glob("**/*.html"))
        for html_file in html_files:
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # Check for CDN links
                    cdn_patterns = [
                        r'https://cdn\.jsdelivr\.net',
                        r'https://cdnjs\.cloudflare\.com',
                        r'https://unpkg\.com'
                    ]

                    for pattern in cdn_patterns:
                        if re.search(pattern, content):
                            # Check for integrity attributes
                            if 'integrity=' not in content:
                                warning = {
                                    'type': 'MISSING_INTEGRITY',
                                    'severity': 'MEDIUM',
                                    'file': str(html_file.relative_to(project_root)),
                                    'line': 0,
                                    'code': 'CDN link without integrity',
                                    'description': 'CDN resource without integrity attribute',
                                    'recommendation': 'Add integrity attributes to CDN resources'
                                }
                                self.audit_results['warnings'].append(warning)

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error reading {html_file}: {e}")

    def check_file_permissions(self):
        """Check file permissions for sensitive files"""
        print("🔍 Checking file permissions...")

        sensitive_files = [
            'config.js',
            'api-key.js',
            'secrets.js',
            '.env'
        ]

        for file_name in sensitive_files:
            file_path = self.website_path / file_name
            if file_path.exists():
                try:
                    stat = file_path.stat()
                    # Check if file is readable by others
                    if stat.st_mode & 0o044:
                        warning = {
                            'type': 'FILE_PERMISSIONS',
                            'severity': 'MEDIUM',
                            'file': str(file_path.relative_to(project_root)),
                            'line': 0,
                            'code': f'Permissions: {oct(stat.st_mode)[-3:]}',
                            'description': 'Sensitive file with overly permissive permissions',
                            'recommendation': 'Restrict file permissions to owner only'
                        }
                        self.audit_results['warnings'].append(warning)

                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Error checking permissions for {file_path}: {e}")

    def check_sensitive_data_exposure(self):
        """Check for sensitive data exposure in client-side code"""
        print("🔍 Checking for sensitive data exposure...")

        js_files = list(self.website_path.glob("**/*.js"))
        sensitive_patterns = [
            r'password\s*[:=]',
            r'secret\s*[:=]',
            r'api[_-]?key\s*[:=]',
            r'token\s*[:=]',
            r'private[_-]?key\s*[:=]'
        ]

        for js_file in js_files:
            try:
                with open(js_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                    for i, line in enumerate(lines, 1):
                        for pattern in sensitive_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                # Check if it's a hardcoded value
                                if re.search(r'["\'][^"\']+["\']', line):
                                    vulnerability = {
                                        'type': 'SENSITIVE_DATA_EXPOSURE',
                                        'severity': 'HIGH',
                                        'file': str(js_file.relative_to(project_root)),
                                        'line': i,
                                        'code': line.strip(),
                                        'description': 'Potential sensitive data exposure in client-side code',
                                        'recommendation': 'Move sensitive data to server-side or use environment variables'
                                    }
                                    self.audit_results['vulnerabilities'].append(vulnerability)

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error reading {js_file}: {e}")

    def check_authentication_security(self):
        """Check authentication and authorization security"""
        print("🔍 Checking authentication security...")

        js_files = list(self.website_path.glob("**/*.js"))
        auth_patterns = [
            r'localStorage\.setItem',
            r'sessionStorage\.setItem',
            r'cookie\s*=',
            r'jwt\s*[:=]',
            r'token\s*[:=]'
        ]

        for js_file in js_files:
            try:
                with open(js_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')

                    for i, line in enumerate(lines, 1):
                        for pattern in auth_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                # Check for secure storage practices
                                if 'localStorage' in line and 'token' in line.lower():
                                    warning = {
                                        'type': 'INSECURE_TOKEN_STORAGE',
                                        'severity': 'MEDIUM',
                                        'file': str(js_file.relative_to(project_root)),
                                        'line': i,
                                        'code': line.strip(),
                                        'description': 'Token stored in localStorage (not secure)',
                                        'recommendation': 'Use httpOnly cookies or secure session storage'
                                    }
                                    self.audit_results['warnings'].append(warning)

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error reading {js_file}: {e}")

    def calculate_security_score(self):
        """Calculate overall security score"""
        total_issues = len(self.audit_results['vulnerabilities']) + len(self.audit_results['warnings'])
        high_severity = len([v for v in self.audit_results['vulnerabilities'] if v['severity'] == 'HIGH'])
        medium_severity = len([v for v in self.audit_results['vulnerabilities'] if v['severity'] == 'MEDIUM'])

        # Calculate score (100 - penalties)
        score = 100
        score -= high_severity * 20  # -20 points per high severity issue
        score -= medium_severity * 10  # -10 points per medium severity issue
        score -= len(self.audit_results['warnings']) * 5  # -5 points per warning

        self.audit_results['security_score'] = max(0, score)

        # Add summary
        self.audit_results['summary'] = {
            'total_vulnerabilities': len(self.audit_results['vulnerabilities']),
            'total_warnings': len(self.audit_results['warnings']),
            'high_severity': high_severity,
            'medium_severity': medium_severity,
            'security_score': self.audit_results['security_score'],
            'security_rating': self.get_security_rating(self.audit_results['security_score'])
        }

    @staticmethod
    def get_security_rating(score):
        """Get security rating based on score"""
        if score >= 90:
            return 'EXCELLENT'
        if score >= 80:
            return 'GOOD'
        if score >= 70:
            return 'FAIR'
        if score >= 60:
            return 'POOR'
        return 'CRITICAL'

    def generate_security_report(self):
        """Generate comprehensive security report"""
        print("\n" + "=" * 60)
        print("🔒 SECURITY AUDIT REPORT")
        print("=" * 60)

        summary = self.audit_results['summary']
        print(f"📊 Security Score: {summary['security_score']}/100 ({summary['security_rating']})")
        print(f"🚨 Vulnerabilities: {summary['total_vulnerabilities']} (High: {summary['high_severity']}, Medium: {summary['medium_severity']})")
        print(f"⚠️  Warnings: {summary['total_warnings']}")

        # Print vulnerabilities
        if self.audit_results['vulnerabilities']:
            print("\n🚨 VULNERABILITIES:")
            for vuln in self.audit_results['vulnerabilities']:
                print(f"  [{vuln['severity']}] {vuln['type']} in {vuln['file']}:{vuln['line']}")
                print(f"    {vuln['description']}")
                print(f"    Recommendation: {vuln['recommendation']}")
                print()

        # Print warnings
        if self.audit_results['warnings']:
            print("\n⚠️  WARNINGS:")
            for warning in self.audit_results['warnings']:
                print(f"  [{warning['severity']}] {warning['type']} in {warning['file']}:{warning['line']}")
                print(f"    {warning['description']}")
                print(f"    Recommendation: {warning['recommendation']}")
                print()

        # Generate recommendations
        self.generate_recommendations()

        # Save report
        self.save_security_report()

        print("=" * 60)
        if summary['security_score'] >= 80:
            print("✅ Security audit completed. Demo website has good security posture.")
        elif summary['security_score'] >= 60:
            print("⚠️  Security audit completed. Some issues need attention.")
        else:
            print("❌ Security audit completed. Critical security issues found.")
        print("=" * 60)

    def generate_recommendations(self):
        """Generate security recommendations"""
        recommendations = []

        # Check for common issues and generate recommendations
        vuln_types = [v['type'] for v in self.audit_results['vulnerabilities']]
        warning_types = [w['type'] for w in self.audit_results['warnings']]

        if 'XSS' in vuln_types:
            recommendations.append({
                'priority': 'HIGH',
                'title': 'Implement XSS Protection',
                'description': 'Add input validation and output encoding to prevent XSS attacks',
                'actions': [
                    'Use textContent instead of innerHTML',
                    'Implement input sanitization',
                    'Add Content Security Policy headers',
                    'Validate and escape user input'
                ]
            })

        if 'INSECURE_HTTP' in vuln_types:
            recommendations.append({
                'priority': 'HIGH',
                'title': 'Enforce HTTPS',
                'description': 'Ensure all API communications use HTTPS',
                'actions': [
                    'Replace HTTP URLs with HTTPS',
                    'Implement HSTS headers',
                    'Use secure API endpoints only'
                ]
            })

        if 'MISSING_CSP' in warning_types:
            recommendations.append({
                'priority': 'MEDIUM',
                'title': 'Implement Content Security Policy',
                'description': 'Add CSP headers to prevent various attacks',
                'actions': [
                    'Add CSP meta tags to HTML files',
                    'Configure CSP directives',
                    'Test CSP implementation'
                ]
            })

        if 'VULNERABLE_DEPENDENCY' in warning_types:
            recommendations.append({
                'priority': 'HIGH',
                'title': 'Update Dependencies',
                'description': 'Update vulnerable dependencies to secure versions',
                'actions': [
                    'Run npm audit fix',
                    'Update to latest versions',
                    'Remove unused dependencies'
                ]
            })

        self.audit_results['recommendations'] = recommendations

        if recommendations:
            print("\n💡 SECURITY RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"\n  [{rec['priority']}] {rec['title']}")
                print(f"    {rec['description']}")
                print("    Actions:")
                for action in rec['actions']:
                    print(f"      - {action}")

    def save_security_report(self):
        """Save security report to file"""
        report_file = project_root / "artifacts" / "security-reports" / f"demo_security_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(self.audit_results, f, indent=2)

        print(f"\n📄 Security report saved to: {report_file}")

        # Generate HTML report if requested
        if self.output_format == 'html':
            self.generate_html_report(report_file)

    def generate_html_report(self, json_file):
        """Generate HTML security report"""
        html_file = json_file.with_suffix('.html')

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Security Audit Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; }}
        .score {{ font-size: 24px; font-weight: bold; }}
        .excellent {{ color: #28a745; }}
        .good {{ color: #17a2b8; }}
        .fair {{ color: #ffc107; }}
        .poor {{ color: #fd7e14; }}
        .critical {{ color: #dc3545; }}
        .vulnerability {{ background: #f8d7da; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .warning {{ background: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .recommendation {{ background: #d1ecf1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔒 Security Audit Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <div class="score {self.get_security_rating(self.audit_results['security_score']).lower()}">
            Security Score: {self.audit_results['security_score']}/100 ({self.audit_results['summary']['security_rating']})
        </div>
    </div>
    
    <h2>Summary</h2>
    <ul>
        <li>Total Vulnerabilities: {self.audit_results['summary']['total_vulnerabilities']}</li>
        <li>High Severity: {self.audit_results['summary']['high_severity']}</li>
        <li>Medium Severity: {self.audit_results['summary']['medium_severity']}</li>
        <li>Warnings: {self.audit_results['summary']['total_warnings']}</li>
    </ul>
    
    <h2>Vulnerabilities</h2>
    {self._generate_html_vulnerabilities()}
    
    <h2>Warnings</h2>
    {self._generate_html_warnings()}
    
    <h2>Recommendations</h2>
    {self._generate_html_recommendations()}
</body>
</html>
        """

        with open(html_file, 'w') as f:
            f.write(html_content)

        print(f"📄 HTML report saved to: {html_file}")

    def _generate_html_vulnerabilities(self):
        """Generate HTML for vulnerabilities section"""
        if not self.audit_results['vulnerabilities']:
            return "<p>No vulnerabilities found.</p>"

        html = ""
        for vuln in self.audit_results['vulnerabilities']:
            html += f"""
            <div class="vulnerability">
                <h3>[{vuln['severity']}] {vuln['type']}</h3>
                <p><strong>File:</strong> {vuln['file']}:{vuln['line']}</p>
                <p><strong>Description:</strong> {vuln['description']}</p>
                <p><strong>Code:</strong> <code>{vuln['code']}</code></p>
                <p><strong>Recommendation:</strong> {vuln['recommendation']}</p>
            </div>
            """
        return html

    def _generate_html_warnings(self):
        """Generate HTML for warnings section"""
        if not self.audit_results['warnings']:
            return "<p>No warnings found.</p>"

        html = ""
        for warning in self.audit_results['warnings']:
            html += f"""
            <div class="warning">
                <h3>[{warning['severity']}] {warning['type']}</h3>
                <p><strong>File:</strong> {warning['file']}:{warning['line']}</p>
                <p><strong>Description:</strong> {warning['description']}</p>
                <p><strong>Code:</strong> <code>{warning['code']}</code></p>
                <p><strong>Recommendation:</strong> {warning['recommendation']}</p>
            </div>
            """
        return html

    def _generate_html_recommendations(self):
        """Generate HTML for recommendations section"""
        if not self.audit_results['recommendations']:
            return "<p>No recommendations available.</p>"

        html = ""
        for rec in self.audit_results['recommendations']:
            html += f"""
            <div class="recommendation">
                <h3>[{rec['priority']}] {rec['title']}</h3>
                <p><strong>Description:</strong> {rec['description']}</p>
                <p><strong>Actions:</strong></p>
                <ul>
                    {''.join(f'<li>{action}</li>' for action in rec['actions'])}
                </ul>
            </div>
            """
        return html


def main():
    """Main entry point for security audit"""
    parser = argparse.ArgumentParser(description='Run SAMO-DL Demo Website Security Audit')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output-format', '-f', choices=['json', 'html'], default='json', help='Output format')

    args = parser.parse_args()

    # Create security auditor
    auditor = DemoSecurityAuditor(
        verbose=args.verbose,
        output_format=args.output_format
    )

    # Run security audit
    auditor.run_security_audit()


if __name__ == '__main__':
    main()
