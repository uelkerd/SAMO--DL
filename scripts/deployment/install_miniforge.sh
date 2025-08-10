#!/usr/bin/env bash

# Robust Miniforge installer with checksum verification and cross-platform support
# Usage: install_miniforge.sh [--prefix <install_dir>]

set -euo pipefail

PREFIX="${HOME}/miniconda"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"; shift 2;;
    *)
      echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

workdir="$(mktemp -d)"
cleanup() {
  rm -rf "$workdir" || true
}
trap cleanup EXIT INT TERM

detect_os_arch() {
  local uname_s uname_m os_name arch_name
  uname_s="$(uname -s)"
  uname_m="$(uname -m)"
  case "$uname_s" in
    Darwin) os_name="MacOSX" ;;
    Linux)  os_name="Linux" ;;
    *)      os_name="$uname_s" ;;
  esac

  case "$os_name" in
    MacOSX)
      case "$uname_m" in
        arm64|aarch64) arch_name="arm64" ;;
        x86_64)        arch_name="x86_64" ;;
        *)             arch_name="x86_64" ;;
      esac
      ;;
    Linux)
      case "$uname_m" in
        aarch64|arm64) arch_name="aarch64" ;;
        x86_64)        arch_name="x86_64" ;;
        *)             arch_name="x86_64" ;;
      esac
      ;;
    *)
      echo "Unsupported OS detected: ${os_name} (uname -s: ${uname_s})" >&2
      exit 1
      ;;
  esac

  echo "$os_name" "$arch_name"
}

main() {
  read -r OS_NAME ARCH_NAME < <(detect_os_arch)

  local BASENAME="Miniforge3-${OS_NAME}-${ARCH_NAME}.sh"
  local LATEST_URL="https://github.com/conda-forge/miniforge/releases/latest/download/${BASENAME}"

  echo "Resolved installer: ${BASENAME}"

  # Use 'latest/download' endpoint for both installer and checksum (avoid API and tag parsing)
  local CHECKSUM_URL="https://github.com/conda-forge/miniforge/releases/latest/download/${BASENAME}.sha256"
  echo "Checksum URL: ${CHECKSUM_URL}"

  # Download installer and checksum (support GitHub token to reduce rate limits)
  local AUTH_HEADER=""
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    AUTH_HEADER="Authorization: Bearer ${GITHUB_TOKEN}"
  fi

  if command -v curl >/dev/null 2>&1; then
    if [[ -n "${AUTH_HEADER}" ]]; then
      curl -sSfL -H "${AUTH_HEADER}" "$LATEST_URL" -o "${workdir}/miniforge.sh"
      curl -sSfL -H "${AUTH_HEADER}" "$CHECKSUM_URL" -o "${workdir}/miniforge.sha256"
    else
      curl -sSfL "$LATEST_URL" -o "${workdir}/miniforge.sh"
      curl -sSfL "$CHECKSUM_URL" -o "${workdir}/miniforge.sha256"
    fi
  elif command -v wget >/dev/null 2>&1; then
    if [[ -n "${AUTH_HEADER}" ]]; then
      wget --header="${AUTH_HEADER}" -qO "${workdir}/miniforge.sh" "$LATEST_URL"
      wget --header="${AUTH_HEADER}" -qO "${workdir}/miniforge.sha256" "$CHECKSUM_URL"
    else
      wget -qO "${workdir}/miniforge.sh" "$LATEST_URL"
      wget -qO "${workdir}/miniforge.sha256" "$CHECKSUM_URL"
    fi
  else
    echo "Error: neither curl nor wget available to download Miniforge." >&2
    exit 1
  fi

  # Verify checksum (support macOS shasum fallback)
  local ACTUAL EXPECTED
  if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL=$(sha256sum "${workdir}/miniforge.sh" | awk '{print $1}')
  else
    ACTUAL=$(shasum -a 256 "${workdir}/miniforge.sh" | awk '{print $1}')
  fi
  EXPECTED=$(awk '{print $1}' < "${workdir}/miniforge.sha256")

  echo "Actual checksum:   ${ACTUAL}"
  echo "Expected checksum: ${EXPECTED}"

  if [[ -z "${EXPECTED}" ]] || [[ "${ACTUAL}" != "${EXPECTED}" ]]; then
    echo "Checksum verification failed or missing. Aborting." >&2
    exit 1
  fi

  # Install Miniforge
  chmod +x "${workdir}/miniforge.sh"
  bash "${workdir}/miniforge.sh" -b -p "${PREFIX}"
  echo "Installed Miniforge at: ${PREFIX}"
}

main "$@"

