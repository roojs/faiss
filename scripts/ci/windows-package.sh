#!/usr/bin/env bash
# Pack the staged Faiss install as an MSYS2 UCRT64 pacman package.
#
# Expects scripts/ci/windows-msys2-build.sh to have filled
# packaging/msys2/prebuilt/ucrt64. Same makepkg-mingw call OLLMchat uses
# for its library packages.
#
# Usage (MSYS2 UCRT64):
#   ./scripts/ci/windows-package.sh
#
# Output:
#   packaging/msys2/mingw-w64-ucrt-x86_64-faiss-<version>-1-any.pkg.tar.zst
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PKG_DIR="${ROOT}/packaging/msys2"
STAGE="${FAISS_PREBUILT_DESTDIR:-${PKG_DIR}/prebuilt}"
MAKEPKG="${MAKEPKG:-makepkg-mingw}"

if [[ "${MSYSTEM:-}" != "UCRT64" ]]; then
  echo "windows-package: MSYSTEM=${MSYSTEM:-unset}; run under MSYS2 UCRT64" >&2
  exit 1
fi
if ! command -v "${MAKEPKG}" >/dev/null 2>&1; then
  echo "windows-package: ${MAKEPKG} not found (MSYS2 UCRT64; pacman -S base-devel)" >&2
  exit 1
fi
if [[ -z "${FAISS_PKGVER:-}" && -f "${PKG_DIR}/buildenv.sh" ]]; then
  # shellcheck disable=SC1091
  source "${PKG_DIR}/buildenv.sh"
  if [[ -n "${FAISS_PREBUILT_DESTDIR:-}" ]]; then
    STAGE="${FAISS_PREBUILT_DESTDIR}"
  fi
fi
if [[ -z "${FAISS_PKGVER:-}" ]]; then
  FAISS_PKGVER="$(sed -n 's/^[[:space:]]*VERSION[[:space:]]\{1,\}\([0-9][0-9.]*\).*/\1/p' CMakeLists.txt | head -1)"
fi
FAISS_PKGVER="${FAISS_PKGVER#v}"
FAISS_PKGVER="${FAISS_PKGVER%-win}"
if [[ -z "${FAISS_PKGVER}" ]]; then
  echo "windows-package: FAISS_PKGVER is not set" >&2
  exit 1
fi
export FAISS_PKGVER
export FAISS_PREBUILT_DESTDIR="${STAGE}"

if [[ ! -f "${STAGE}/ucrt64/bin/libfaiss.dll" ]]; then
  echo "windows-package: ${STAGE}/ucrt64/bin/libfaiss.dll missing (run windows-msys2-build.sh)" >&2
  exit 1
fi

cat > "${PKG_DIR}/buildenv.sh" <<EOF
FAISS_PKGVER='${FAISS_PKGVER}'
FAISS_PREBUILT_DESTDIR='${STAGE}'
EOF

rm -f "${PKG_DIR}"/mingw-w64-ucrt-x86_64-faiss-*-any.pkg.tar.zst \
  "${PKG_DIR}"/mingw-w64-ucrt-x86_64-faiss-*-any.pkg.tar.zst.sig

echo "==> ${MAKEPKG} faiss ${FAISS_PKGVER}"
(
  cd "${PKG_DIR}"
  "${MAKEPKG}" -C -f --noconfirm --skipchecksums
)

pkg="${PKG_DIR}/mingw-w64-ucrt-x86_64-faiss-${FAISS_PKGVER}-1-any.pkg.tar.zst"
if [[ ! -f "${pkg}" ]]; then
  echo "windows-package: missing ${pkg}" >&2
  ls -l "${PKG_DIR}"/*.pkg.tar.zst >&2 || true
  exit 1
fi
ls -lh "${pkg}"
echo "==> done"
