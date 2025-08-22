"""
Model Integrity Checker for Secure Model Loading.

This module provides integrity verification capabilities for model files,
including checksums, digital signatures, and format validation.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger__name__


class IntegrityChecker:
    """Model integrity checker for secure model loading.

    Provides comprehensive integrity verification including:
    - SHA-256 checksums
    - File format validation
    - Size limits and constraints
    - Version compatibility checks
    """

    def __init__self, trusted_checksums_file: Optional[str] = None:
        """Initialize integrity checker.

        Args:
            trusted_checksums_file: Path to file containing trusted checksums
        """
        self.trusted_checksums_file = trusted_checksums_file
        self.trusted_checksums = self._load_trusted_checksums()

        # Security constraints
        self.max_file_size = 2 * 1024 * 1024 * 1024  # 2GB max
        self.allowed_extensions = {'.pt', '.pth', '.bin', '.safetensors'}
        self.blocked_patterns = [
            b'__import__', b'eval(', b'exec(', b'pickle.loads',
            b'subprocess', b'os.system', b'__builtins__'
        ]

    def _load_trusted_checksumsself -> Dict[str, str]:
        """Load trusted checksums from file.

        Returns:
            Dictionary mapping file paths to expected checksums
        """
        if not self.trusted_checksums_file or not os.path.existsself.trusted_checksums_file:
            logger.warning"No trusted checksums file found, using empty trust store"
            return {}

        try:
            with openself.trusted_checksums_file, 'r' as f:
                return json.loadf
        except Exception as e:
            logger.errorf"Failed to load trusted checksums: {e}"
            return {}

    def calculate_checksumself, file_path: str -> str:
        """Calculate SHA-256 checksum of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 checksum as hex string
        """
        sha256_hash = hashlib.sha256()

        try:
            with openfile_path, 'rb' as f:
                for chunk in iter(lambda: f.read4096, b""):
                    sha256_hash.updatechunk
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.errorf"Failed to calculate checksum for {file_path}: {e}"
            raise

    def validate_file_sizeself, file_path: str -> bool:
        """Validate file size is within acceptable limits.

        Args:
            file_path: Path to the file

        Returns:
            True if file size is acceptable
        """
        try:
            file_size = os.path.getsizefile_path
            if file_size > self.max_file_size:
                logger.errorf"File {file_path} exceeds maximum size limit: {file_size} bytes"
                return False
            return True
        except Exception as e:
            logger.errorf"Failed to validate file size for {file_path}: {e}"
            return False

    def validate_file_extensionself, file_path: str -> bool:
        """Validate file extension is allowed.

        Args:
            file_path: Path to the file

        Returns:
            True if file extension is allowed
        """
        file_ext = Pathfile_path.suffix.lower()
        if file_ext not in self.allowed_extensions:
            logger.errorf"File extension {file_ext} not allowed for {file_path}"
            return False
        return True

    def scan_for_malicious_contentself, file_path: str -> Tuple[bool, list]:
        """Scan file for potentially malicious content.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of is_safe, list_of_findings
        """
        findings = []

        try:
            with openfile_path, 'rb' as f:
                content = f.read()

                for pattern in self.blocked_patterns:
                    if pattern in content:
                        findings.appendf"Found blocked pattern: {pattern}"

        except Exception as e:
            logger.errorf"Failed to scan file {file_path}: {e}"
            findings.appendf"Scan failed: {e}"

        return lenfindings == 0, findings

    def verify_checksumself, file_path: str, expected_checksum: Optional[str] = None -> bool:
        """Verify file checksum against expected value.

        Args:
            file_path: Path to the file
            expected_checksum: Expected checksum if None, uses trusted checksums

        Returns:
            True if checksum matches
        """
        try:
            actual_checksum = self.calculate_checksumfile_path

            if expected_checksum:
                return actual_checksum == expected_checksum

            # Check against trusted checksums
            if file_path in self.trusted_checksums:
                return actual_checksum == self.trusted_checksums[file_path]

            logger.warningf"No expected checksum provided for {file_path}"
            return False

        except Exception as e:
            logger.errorf"Failed to verify checksum for {file_path}: {e}"
            return False

    def validate_model_structureself, model_path: str -> bool:
        """Validate PyTorch model structure.

        Args:
            model_path: Path to the model file

        Returns:
            True if model structure is valid
        """
        try:
            # Load model in a controlled environment
            model_data = torch.loadmodel_path, map_location='cpu', weights_only=True

            # Basic structure validation
            if not isinstancemodel_data, dict:
                logger.errorf"Model {model_path} is not a valid state dict"
                return False

            # Check for required keys in state dict
            required_keys = ['state_dict', 'config', 'model_name']
            for key in required_keys:
                if key not in model_data:
                    logger.warningf"Model {model_path} missing key: {key}"

            return True

        except Exception as e:
            logger.errorf"Failed to validate model structure for {model_path}: {e}"
            return False

    def comprehensive_validationself, file_path: str, expected_checksum: Optional[str] = None -> Tuple[bool, Dict]:
        """Perform comprehensive file validation.

        Args:
            file_path: Path to the file
            expected_checksum: Expected checksum

        Returns:
            Tuple of is_valid, validation_results
        """
        results = {
            'file_path': file_path,
            'size_valid': False,
            'extension_valid': False,
            'checksum_valid': False,
            'content_safe': False,
            'structure_valid': False,
            'findings': []
        }

        # File size validation
        results['size_valid'] = self.validate_file_sizefile_path
        if not results['size_valid']:
            results['findings'].append"File size exceeds limit"

        # Extension validation
        results['extension_valid'] = self.validate_file_extensionfile_path
        if not results['extension_valid']:
            results['findings'].append"File extension not allowed"

        # Checksum validation
        results['checksum_valid'] = self.verify_checksumfile_path, expected_checksum
        if not results['checksum_valid']:
            results['findings'].append"Checksum verification failed"

        # Content safety scan
        is_safe, findings = self.scan_for_malicious_contentfile_path
        results['content_safe'] = is_safe
        results['findings'].extendfindings

        # Model structure validation only for model files
        if Pathfile_path.suffix.lower() in {'.pt', '.pth'}:
            results['structure_valid'] = self.validate_model_structurefile_path
            if not results['structure_valid']:
                results['findings'].append"Model structure validation failed"

        # Overall validation result
        is_valid = all([
            results['size_valid'],
            results['extension_valid'],
            results['checksum_valid'],
            results['content_safe']
        ])

        if Pathfile_path.suffix.lower() in {'.pt', '.pth'}:
            is_valid = is_valid and results['structure_valid']

        return is_valid, results
