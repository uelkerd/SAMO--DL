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
      arch_name="x86_64"
      ;;
  esac

  echo "$os_name" "$arch_name"
}

main() {
  read -r OS_NAME ARCH_NAME < <(detect_os_arch)

  local BASENAME="Miniforge3-${OS_NAME}-${ARCH_NAME}.sh"
  local LATEST_URL="https://github.com/conda-forge/miniforge/releases/latest/download/${BASENAME}"

  echo "Resolved installer: ${BASENAME}"

  # Fetch latest tag for checksum URL
  curl -sSfL https://api.github.com/repos/conda-forge/miniforge/releases/latest -o "${workdir}/latest_release.json"
  local TAG
  TAG=$(sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "${workdir}/latest_release.json" | head -n1)
  if [[ -z "${TAG}" ]]; then
    echo "Failed to resolve latest tag from GitHub API" >&2
    exit 1
  fi

  local CHECKSUM_URL="https://github.com/conda-forge/miniforge/releases/download/${TAG}/Miniforge3-${TAG}-${OS_NAME}-${ARCH_NAME}.sh.sha256"
  echo "Checksum URL: ${CHECKSUM_URL}"

  # Download installer and checksum
  if command -v curl >/dev/null 2>&1; then
    curl -sSfL "$LATEST_URL" -o "${workdir}/miniforge.sh"
    curl -sSfL "$CHECKSUM_URL" -o "${workdir}/miniforge.sha256"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "${workdir}/miniforge.sh" "$LATEST_URL"
    wget -qO "${workdir}/miniforge.sha256" "$CHECKSUM_URL"
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

