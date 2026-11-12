#!/usr/bin/env bash
# GSL from source, static only, into PREFIX: what the macOS wheels link the C
# engine against (cibuildwheel's before-all, [tool.cibuildwheel.macos]).
#
#   bash tools/build_gsl.sh PREFIX
#
# Why from source: a Homebrew bottle is built for the RUNNER's macOS (15 on
# macos-15), and a wheel tagged for macOS 11 cannot carry it (delocate refuses,
# or worse, the import fails on the user's Mac). Compiled here, clang honours
# MACOSX_DEPLOYMENT_TARGET (11.0, set by the workflow) and the runner's
# architecture (arm64 on macos-15, x86_64 on macos-15-intel). Static, so the
# extension carries GSL inside and delocate has nothing left to bundle.
# GSL 2.8 is the version the Homebrew bottle gave the wheels up to now; the
# tarball is checked against its sha256. tools/check_macos_minos.py checks the
# result. The same script, with GSL 2.7.1, builds drvec-python's binary.
set -euo pipefail
PREFIX=${1:?usage: build_gsl.sh PREFIX}
GSL_VERSION=2.8
GSL_SHA256=6a99eeed15632c6354895b1dd542ed5a855c0f15d9ad1326c6fe2b2c9e423190

if [ -f "$PREFIX/lib/libgsl.a" ] && [ -f "$PREFIX/lib/libgslcblas.a" ]; then
    echo "build_gsl: $PREFIX already has GSL"; exit 0
fi
echo "build_gsl: GSL $GSL_VERSION for macOS ${MACOSX_DEPLOYMENT_TARGET:-?} on $(uname -m)"
WORK=$(mktemp -d)
cd "$WORK"
T=gsl-$GSL_VERSION.tar.gz
for url in "https://ftp.gnu.org/gnu/gsl/$T" "https://mirrors.kernel.org/gnu/gsl/$T"; do
    if curl -fsSL --retry 3 -o "$T" "$url"; then break; fi
done
if command -v sha256sum >/dev/null 2>&1; then
    got=$(sha256sum "$T" | cut -d' ' -f1)
else
    got=$(shasum -a 256 "$T" | cut -d' ' -f1)
fi
if [ "$got" != "$GSL_SHA256" ]; then
    echo "build_gsl: $T has sha256 $got, expected $GSL_SHA256" >&2; exit 1
fi
tar xzf "$T"
cd gsl-$GSL_VERSION
NJ=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
./configure --prefix="$PREFIX" --disable-shared --enable-static CFLAGS="-O2 -fPIC" >/dev/null
make -j"$NJ" >/dev/null
make install >/dev/null
ls -la "$PREFIX/lib/libgsl.a" "$PREFIX/lib/libgslcblas.a"
rm -rf "$WORK"
