#!/usr/bin/env bash
# Native Windows (MSYS2 UCRT64) build for Faiss — used by GitHub Actions
# (.github/workflows/x-windows.yml) and a local UCRT64 shell.
#
# Same shape as OLLMchat scripts/ci/windows-msys2-build.sh: compile, install
# into a DESTDIR, then normalize to packaging/msys2/prebuilt/ucrt64 so pacman
# extracts onto /ucrt64 on the machine that installs the package.
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ "${MSYSTEM:-}" != "UCRT64" ]]; then
  echo "windows-msys2-build: MSYSTEM=${MSYSTEM:-unset}; run under MSYS2 UCRT64" >&2
  exit 1
fi

PREFIX="${MSYSTEM_PREFIX:-/ucrt64}"
BUILD_DIR="${BUILD_DIR:-build-windows-x86_64}"
STAGE="${ROOT}/packaging/msys2/prebuilt"

# FAISS_PKGVER may be "v1.16.0" or "1.16.0". A branch name is not a version.
if [[ -z "${FAISS_PKGVER:-}" && "${GITHUB_REF:-}" == refs/tags/v* ]]; then
  FAISS_PKGVER="${GITHUB_REF#refs/tags/}"
fi
FAISS_PKGVER="${FAISS_PKGVER:-}"
FAISS_PKGVER="${FAISS_PKGVER#v}"
FAISS_PKGVER="${FAISS_PKGVER%-win32}"
FAISS_PKGVER="${FAISS_PKGVER%-win}"
if [[ ! "${FAISS_PKGVER}" =~ ^[0-9] ]]; then
  FAISS_PKGVER="$(sed -n 's/^[[:space:]]*VERSION[[:space:]]\{1,\}\([0-9][0-9.]*\).*/\1/p' CMakeLists.txt | head -1)"
fi
if [[ -z "${FAISS_PKGVER}" ]]; then
  echo "windows-msys2-build: could not determine FAISS_PKGVER" >&2
  exit 1
fi
export FAISS_PKGVER
echo "==> FAISS ${FAISS_PKGVER} -> ${STAGE}"

rm -rf "${BUILD_DIR}" "${STAGE}"
# -fpermissive matches the MinGW build OLLMchat used for Faiss.
cmake -S . -B "${BUILD_DIR}" -G Ninja \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_MKL=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-fpermissive \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}"
cmake --build "${BUILD_DIR}" --target faiss
DESTDIR="${STAGE}" cmake --install "${BUILD_DIR}"

# setup-msys2 lives under a runner-specific root. cmake resolves /ucrt64 to
# that absolute path, so DESTDIR is not yet prebuilt/ucrt64.
dll="$(find "${STAGE}" -path '*/bin/libfaiss.dll' -print -quit || true)"
if [[ -z "${dll}" ]]; then
  echo "windows-msys2-build: libfaiss.dll not under ${STAGE}" >&2
  find "${STAGE}" -type f -print >&2 || true
  exit 1
fi
prefix_root="$(cd "$(dirname "${dll}")/.." && pwd)"
canon="${STAGE}/ucrt64"
if [[ "${prefix_root}" != "${canon}" ]]; then
  staged="${STAGE}/.faiss-ucrt64-stage"
  rm -rf "${staged}"
  mkdir -p "${staged}/ucrt64"
  cp -a "${prefix_root}/." "${staged}/ucrt64/"
  find "${STAGE}" -mindepth 1 -maxdepth 1 ! -name '.faiss-ucrt64-stage' -exec rm -rf {} +
  mv "${staged}/ucrt64" "${canon}"
  rm -rf "${staged}"
fi

install -Dm644 "${ROOT}/LICENSE" \
  "${canon}/share/licenses/mingw-w64-ucrt-x86_64-faiss/LICENSE"

if [[ ! -f "${canon}/bin/libfaiss.dll" ]]; then
  echo "windows-msys2-build: normalize failed (${canon}/bin/libfaiss.dll)" >&2
  exit 1
fi
# makepkg starts a clean environment, and the package step is a second
# process. Leave the version and stage path next to the PKGBUILD.
cat > "${ROOT}/packaging/msys2/buildenv.sh" <<EOF
FAISS_PKGVER='${FAISS_PKGVER}'
FAISS_PREBUILT_DESTDIR='${STAGE}'
EOF

echo "==> staged ${canon}"
find "${canon}" -type f -print | sort
echo "FAISS_PKGVER=${FAISS_PKGVER}"
