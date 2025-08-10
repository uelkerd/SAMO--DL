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
  local LATEST_BASE="https://github.com/conda-forge/miniforge/releases/latest/download"
  local LATEST_URL="${LATEST_BASE}/${BASENAME}"

  echo "Resolved installer: ${BASENAME}"

  # Determine checksum source with fallbacks
  # 1) Per-file checksum ("<asset>.sha256")
  # 2) Release checksum bundle ("SHA256SUMS" or "sha256sum.txt")
  local CHECKSUM_URL1="${LATEST_BASE}/${BASENAME}.sha256"
  local CHECKSUM_URL2="${LATEST_BASE}/SHA256SUMS"
  local CHECKSUM_URL3="${LATEST_BASE}/sha256sum.txt"

  # Download installer and checksum (support GitHub token to reduce rate limits)
  local AUTH_HEADER=""
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    AUTH_HEADER="Authorization: Bearer ${GITHUB_TOKEN}"
  fi

  # Downloader helpers
  download() {
    local url="$1" out="$2"
    if command -v curl >/dev/null 2>&1; then
      if [[ -n "${AUTH_HEADER}" ]]; then
        curl -sSfL -H "${AUTH_HEADER}" "$url" -o "$out"
      else
        curl -sSfL "$url" -o "$out"
      fi
    elif command -v wget >/dev/null 2>&1; then
      if [[ -n "${AUTH_HEADER}" ]]; then
        wget --header="${AUTH_HEADER}" -qO "$out" "$url"
      else
        wget -qO "$out" "$url"
      fi
    else
      return 127
    fi
  }

  # Download installer
  download "$LATEST_URL" "${workdir}/miniforge.sh" || { echo "Failed to download installer $LATEST_URL" >&2; exit 1; }

  # Try per-file checksum first
  if download "$CHECKSUM_URL1" "${workdir}/miniforge.sha256"; then
    CHECKSUM_MODE="single"
  else
    # Try checksum bundle files
    if download "$CHECKSUM_URL2" "${workdir}/miniforge.SHA256SUMS"; then
      CHECKSUM_MODE="bundle2"
    elif download "$CHECKSUM_URL3" "${workdir}/miniforge.sha256sum.txt"; then
      CHECKSUM_MODE="bundle3"
    else
      echo "Failed to retrieve checksum from any known source: ${CHECKSUM_URL1} or ${CHECKSUM_URL2} or ${CHECKSUM_URL3}" >&2
      exit 1
    fi
  fi

  # Verify checksum (support macOS shasum fallback)
  local ACTUAL EXPECTED
  if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL=$(sha256sum "${workdir}/miniforge.sh" | awk '{print $1}')
  else
    ACTUAL=$(shasum -a 256 "${workdir}/miniforge.sh" | awk '{print $1}')
  fi
  local EXPECTED
  case "${CHECKSUM_MODE}" in
    single)
      EXPECTED=$(awk '{print $1}' < "${workdir}/miniforge.sha256")
      ;;
    bundle2)
      # SHA256SUMS format: "<hash>  <filename>"
      EXPECTED=$(grep -E "\s${BASENAME}$" "${workdir}/miniforge.SHA256SUMS" | awk '{print $1}' | head -n1)
      ;;
    bundle3)
      # sha256sum.txt format is the same
      EXPECTED=$(grep -E "\s${BASENAME}$" "${workdir}/miniforge.sha256sum.txt" | awk '{print $1}' | head -n1)
      ;;
    *)
      EXPECTED=""
      ;;
  esac

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

