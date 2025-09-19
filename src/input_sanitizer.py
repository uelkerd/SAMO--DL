#!/usr/bin/env python3
"""
🧹 Input Sanitizer
=================
Comprehensive input sanitization and validation for API security.
"""

import re
import html
import logging
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
import unicodedata

logger = logging.getLogger(__name__)

@dataclass
class SanitizationConfig:
    """Input sanitization configuration."""
    max_text_length: int = 10000
    max_batch_size: int = 100
    allowed_html_tags: set = None
    blocked_patterns: set = None
    enable_xss_protection: bool = True
    enable_sql_injection_protection: bool = True
    enable_path_traversal_protection: bool = True
    enable_command_injection_protection: bool = True
    enable_unicode_normalization: bool = True
    enable_content_type_validation: bool = True

class InputSanitizer:
    """
    Comprehensive input sanitization and validation.

    Features:
    - XSS protection
    - SQL injection protection
    - Path traversal protection
    - Command injection protection
    - Unicode normalization
    - Content type validation
    - Length limits
    - Pattern blocking
    """

    def __init__(self, config: SanitizationConfig):
        self.config = config

        # Initialize default blocked patterns
        if config.blocked_patterns is None:
            config.blocked_patterns = {
                # XSS patterns
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'on\w+\s*=',
                r'<iframe[^>]*>',
                r'<object[^>]*>',
                r'<embed[^>]*>',

                # SQL injection patterns
                r'(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)',
                r'(\b(or|and)\b\s+\d+\s*=\s*\d+)',
                r'(\b(union|select)\b.*?\bfrom\b)',
                r'(\b(insert|update|delete)\b.*?\binto\b)',

                # Path traversal patterns
                r'\.\./',
                r'\.\.\\',
                r'%2e%2e%2f',
                r'%2e%2e%5c',

                # Command injection patterns
                r'(\b(cmd|command|exec|system|eval|exec)\b)',
                r'(\b(popen|subprocess|os\.system)\b)',
                r'(\b(shell|bash|sh|powershell)\b)',
                r'(\b(rm|del|format|mkfs)\b)',

                # Other dangerous patterns
                r'(\b(import|__import__)\b)',
                r'(\b(eval|exec|compile)\b)',
                r'(\b(open|file|read|write)\b)',
                r'(\b(subprocess|multiprocessing)\b)',
            }

        # Initialize allowed HTML tags
        if config.allowed_html_tags is None:
            config.allowed_html_tags = {
                'p', 'br', 'strong', 'em', 'u', 'i', 'b', 'span', 'div'
            }

    def sanitize_text(self, text: str, context: str = "general") -> Tuple[str, List[str]]:
        """
        Sanitize text input.

        Args:
            text: Input text to sanitize
            context: Context for sanitization (e.g., "emotion", "general")

        Returns:
            Tuple of (sanitized_text, warnings)
        """
        warnings = []

        if not isinstance(text, str):
            raise ValueError(f"Input must be a string, got {type(text)}")

        # Check length
        if len(text) > self.config.max_text_length:
            warnings.append(f"Text truncated from {len(text)} to {self.config.max_text_length} characters")
            text = text[:self.config.max_text_length]

        # Unicode normalization
        if self.config.enable_unicode_normalization:
            text = unicodedata.normalize('NFKC', text)

        # Check for blocked patterns
        if self.config.enable_xss_protection or self.config.enable_sql_injection_protection:
            for pattern in self.config.blocked_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    warnings.append(f"Blocked pattern detected: {pattern}")
                    # Replace with safe alternative
                    text = re.sub(pattern, '[BLOCKED]', text, flags=re.IGNORECASE)

        # HTML escaping for XSS protection
        if self.config.enable_xss_protection:
            text = html.escape(text)

        # Remove null bytes and control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')

        # Strip leading/trailing whitespace
        text = text.strip()

        return text, warnings

    def sanitize_json(self, data: Any, max_depth: int = 10) -> Tuple[Any, List[str]]:
        """
        Sanitize JSON data recursively.

        Args:
            data: JSON data to sanitize
            max_depth: Maximum recursion depth

        Returns:
            Tuple of (sanitized_data, warnings)
        """
        warnings = []

        def _sanitize_recursive(obj: Any, depth: int = 0) -> Any:
            if depth > max_depth:
                warnings.append(f"Maximum recursion depth {max_depth} exceeded")
                return None

            if isinstance(obj, str):
                sanitized, obj_warnings = self.sanitize_text(obj)
                warnings.extend(obj_warnings)
                return sanitized
            elif isinstance(obj, dict):
                return {k: _sanitize_recursive(v, depth + 1) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_sanitize_recursive(item, depth + 1) for item in obj]
            elif isinstance(obj, (int, float, bool, type(None))):
                return obj
            else:
                warnings.append(f"Unsupported type {type(obj)} converted to string")
                return str(obj)

        return _sanitize_recursive(data), warnings

    def validate_emotion_request(self, data: Dict) -> Tuple[Dict, List[str]]:
        """
        Validate and sanitize emotion detection request.

        Args:
            data: Request data

        Returns:
            Tuple of (sanitized_data, warnings)
        """
        warnings = []
        sanitized_data = {}

        # Validate text field
        if 'text' not in data:
            raise ValueError("Missing required field 'text'")

        text = data['text']
        if not isinstance(text, str):
            raise ValueError("Field 'text' must be a string")

        sanitized_text, text_warnings = self.sanitize_text(text, "emotion")
        sanitized_data['text'] = sanitized_text
        warnings.extend(text_warnings)

        # Validate optional fields
        if 'confidence_threshold' in data:
            try:
                threshold = float(data['confidence_threshold'])
                if 0.0 <= threshold <= 1.0:
                    sanitized_data['confidence_threshold'] = threshold
                else:
                    warnings.append("confidence_threshold must be between 0.0 and 1.0")
            except (ValueError, TypeError):
                warnings.append("confidence_threshold must be a number")

        return sanitized_data, warnings

    def validate_batch_request(self, data: Dict) -> Tuple[Dict, List[str]]:
        """
        Validate and sanitize batch emotion detection request.

        Args:
            data: Request data

        Returns:
            Tuple of (sanitized_data, warnings)
        """
        warnings = []
        sanitized_data = {}

        # Validate texts field
        if 'texts' not in data:
            raise ValueError("Missing required field 'texts'")

        texts = data['texts']
        if not isinstance(texts, list):
            raise ValueError("Field 'texts' must be a list")

        # Check batch size
        if len(texts) > self.config.max_batch_size:
            warnings.append(f"Batch size {len(texts)} exceeds maximum {self.config.max_batch_size}")
            texts = texts[:self.config.max_batch_size]

        # Sanitize each text
        sanitized_texts = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                warnings.append(f"Text at index {i} is not a string, skipping")
                continue

            sanitized_text, text_warnings = self.sanitize_text(text, "emotion")
            sanitized_texts.append(sanitized_text)
            warnings.extend([f"Text {i}: {w}" for w in text_warnings])

        sanitized_data['texts'] = sanitized_texts

        # Validate optional fields
        if 'confidence_threshold' in data:
            try:
                threshold = float(data['confidence_threshold'])
                if 0.0 <= threshold <= 1.0:
                    sanitized_data['confidence_threshold'] = threshold
                else:
                    warnings.append("confidence_threshold must be between 0.0 and 1.0")
            except (ValueError, TypeError):
                warnings.append("confidence_threshold must be a number")

        return sanitized_data, warnings

    def validate_content_type(self, content_type: str) -> bool:
        """
        Validate content type header.

        Args:
            content_type: Content type header value

        Returns:
            True if valid, False otherwise
        """
        if not self.config.enable_content_type_validation:
            return True

        # Check for JSON content type
        if not content_type or 'application/json' not in content_type.lower():
            return False

        return True

    def sanitize_headers(self, headers: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
        """
        Sanitize HTTP headers.

        Args:
            headers: HTTP headers

        Returns:
            Tuple of (sanitized_headers, warnings)
        """
        warnings = []
        sanitized_headers = {}

        for key, value in headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                warnings.append(f"Invalid header type: {key}")
                continue

            # Sanitize header name and value
            sanitized_key, key_warnings = self.sanitize_text(key, "header")
            sanitized_value, value_warnings = self.sanitize_text(value, "header")

            sanitized_headers[sanitized_key] = sanitized_value
            warnings.extend(key_warnings)
            warnings.extend(value_warnings)

        return sanitized_headers, warnings

    def detect_anomalies(self, data: Any) -> List[str]:
        """
        Detect potential security anomalies in data.

        Args:
            data: Data to analyze

        Returns:
            List of detected anomalies
        """
        anomalies = []

        def _analyze_recursive(obj: Any, path: str = ""):
            if isinstance(obj, str):
                # Check for suspicious patterns
                if len(obj) > 1000:
                    anomalies.append(f"Large string at {path}: {len(obj)} characters")

                if re.search(r'[<>"\']', obj):
                    anomalies.append(f"Potential HTML/script content at {path}")

                if re.search(r'\b(union|select|insert|update|delete)\b', obj, re.IGNORECASE):
                    anomalies.append(f"Potential SQL injection at {path}")

            elif isinstance(obj, dict):
                for key, value in obj.items():
                    _analyze_recursive(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    _analyze_recursive(item, f"{path}[{i}]")

        _analyze_recursive(data)
        return anomalies

    def get_sanitization_stats(self) -> Dict:
        """Get sanitization statistics."""
        return {
            "config": {
                "max_text_length": self.config.max_text_length,
                "max_batch_size": self.config.max_batch_size,
                "enable_xss_protection": self.config.enable_xss_protection,
                "enable_sql_injection_protection": self.config.enable_sql_injection_protection,
                "enable_path_traversal_protection": self.config.enable_path_traversal_protection,
                "enable_command_injection_protection": self.config.enable_command_injection_protection,
                "enable_unicode_normalization": self.config.enable_unicode_normalization,
                "enable_content_type_validation": self.config.enable_content_type_validation,
            },
            "blocked_patterns_count": len(self.config.blocked_patterns),
            "allowed_html_tags_count": len(self.config.allowed_html_tags)
        }
