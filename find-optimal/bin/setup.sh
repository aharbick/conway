#!/usr/bin/env bash
# Dev environment setup for Debian/Ubuntu (apt) and Arch/Omarchy (pacman).
#
# Usage:
#   ./setup.sh                      # toolchain, libraries, NVIDIA driver, CUDA toolkit
#   ./setup.sh --no-cuda            # skip the CUDA toolkit (e.g. no NVIDIA GPU)
#   ./setup.sh --no-nvidia-driver   # skip the NVIDIA driver (apt only; skipped if one already works)
set -euo pipefail

WANT_CUDA=1
WANT_DRIVER=1
NVIDIA_DRIVER_VERSION="${NVIDIA_DRIVER_VERSION:-570}"   # apt only

for arg in "$@"; do
  case "$arg" in
    --no-cuda)          WANT_CUDA=0 ;;
    --no-nvidia-driver) WANT_DRIVER=0 ;;
    -h|--help)          sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

# Use sudo only when not already root
SUDO=""
[[ $EUID -ne 0 ]] && SUDO="sudo"

log() { printf '\n==> %s\n' "$*"; }

is_wsl() { grep -qi microsoft /proc/version 2>/dev/null; }

have_nvidia_driver() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; }

# ---------------------------------------------------------------- apt ----
install_apt() {
  export DEBIAN_FRONTEND=noninteractive
  local pkgs=(
    ca-certificates curl
    pkgconf build-essential cmake
    libcurl4-openssl-dev libssl-dev
    clang-format direnv
    libgtest-dev libgmock-dev
    nlohmann-json3-dev
  )

  log "Updating apt and installing packages"
  $SUDO apt-get update
  $SUDO apt-get install -y "${pkgs[@]}"

  if (( WANT_DRIVER )); then
    if is_wsl; then
      log "Skipping NVIDIA driver: WSL uses the Windows host driver"
    elif have_nvidia_driver; then
      log "Skipping NVIDIA driver: one is already installed and working"
    else
      log "Installing nvidia-driver-${NVIDIA_DRIVER_VERSION}"
      $SUDO apt-get install -y "nvidia-driver-${NVIDIA_DRIVER_VERSION}"
    fi
  fi

  if (( WANT_CUDA )); then
    [[ "$(uname -m)" == "x86_64" ]] || { echo "CUDA repo setup here assumes x86_64" >&2; exit 1; }

    local repo
    if is_wsl; then
      repo="wsl-ubuntu"
    else
      # shellcheck disable=SC1091
      . /etc/os-release
      [[ "${ID:-}" == "ubuntu" ]] || { echo "CUDA repo setup supports Ubuntu (found: ${ID:-unknown})" >&2; exit 1; }
      repo="ubuntu${VERSION_ID//./}"       # e.g. 24.04 -> ubuntu2404
    fi

    log "Installing CUDA toolkit from NVIDIA repo ($repo)"
    local deb tmp
    tmp="$(mktemp -d)"
    deb="$tmp/cuda-keyring.deb"
    curl -fsSL -o "$deb" \
      "https://developer.download.nvidia.com/compute/cuda/repos/${repo}/x86_64/cuda-keyring_1.1-1_all.deb"
    $SUDO dpkg -i "$deb"
    rm -rf "$tmp"
    $SUDO apt-get update
    $SUDO apt-get install -y cuda-toolkit
  fi
}

# ------------------------------------------------------------- pacman ----
install_pacman() {
  # Arch ships headers in the main packages (no -dev split);
  # clang-format is part of clang, gmock is part of gtest.
  local pkgs=(
    base-devel cmake
    curl openssl
    clang direnv
    gtest nlohmann-json
  )
  (( WANT_CUDA )) && pkgs+=(cuda)

  if command -v omarchy >/dev/null 2>&1; then
    # Omarchy blocks direct `pacman -Syu`; system upgrades go through `omarchy update`.
    # Install without -y so we use the package database from the last update
    # (avoids a partial upgrade).
    log "Omarchy detected: installing without a system upgrade"
    if ! $SUDO pacman -S --needed --noconfirm "${pkgs[@]}"; then
      echo >&2
      echo "Install failed. If pacman couldn't download packages, the package list is" >&2
      echo "probably stale: run 'omarchy update', then rerun this script." >&2
      exit 1
    fi
  else
    log "Updating system and installing packages"
    # Arch doesn't support partial upgrades, so sync + upgrade + install together
    $SUDO pacman -Syu --needed --noconfirm "${pkgs[@]}"
  fi

  if (( WANT_DRIVER )) && ! have_nvidia_driver; then
    log "No working NVIDIA driver found. Arch has several driver packages (nvidia-open, legacy"
    echo "    branches, ...) and the right one depends on your GPU, so install it yourself:"
    echo "    https://wiki.archlinux.org/title/NVIDIA"
  fi
  if (( WANT_CUDA )); then
    log "CUDA installed to /opt/cuda; open a new login shell for nvcc to be on PATH"
  fi
}

# ----------------------------------------------------------- common ----
hook_direnv() {
  local rc="$HOME/.bashrc"
  local line='eval "$(direnv hook bash)"'
  if [[ -f "$rc" ]] && grep -qF "$line" "$rc"; then
    return
  fi
  log "Adding direnv hook to $rc"
  printf '\n# direnv\n%s\n' "$line" >> "$rc"
}

# ------------------------------------------------------------- main ----
if command -v pacman >/dev/null 2>&1; then
  install_pacman
elif command -v apt-get >/dev/null 2>&1; then
  install_apt
else
  echo "No supported package manager found (need apt or pacman)" >&2
  exit 1
fi

hook_direnv
log "Done. Run 'source ~/.bashrc' or open a new terminal."
