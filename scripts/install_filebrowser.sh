#!/usr/bin/env bash
set -Eeuo pipefail

# Official File Browser v2.63.23, Linux AMD64. Keep the release archive's
# published SHA256 pinned rather than following the latest release at build.
# https://github.com/filebrowser/filebrowser/releases/tag/v2.63.23
FILEBROWSER_INSTALL_VERSION=2.63.23
FILEBROWSER_ARCHIVE_SHA256=b14db2bb8033caa3f80205eb6578b2ed0744ebd9e716b790bc4a9703ce909e88
FILEBROWSER_INSTALL_DIR="$(mktemp -d /tmp/fibo-filebrowser-install.XXXXXX)"
trap 'rm -rf -- "$FILEBROWSER_INSTALL_DIR"' EXIT

curl --fail --location --silent --show-error --proto '=https' --proto-redir '=https' \
  --retry 3 --output "$FILEBROWSER_INSTALL_DIR/filebrowser.tar.gz" \
  "https://github.com/filebrowser/filebrowser/releases/download/v${FILEBROWSER_INSTALL_VERSION}/linux-amd64-filebrowser.tar.gz"
printf '%s  %s\n' "$FILEBROWSER_ARCHIVE_SHA256" "$FILEBROWSER_INSTALL_DIR/filebrowser.tar.gz" | sha256sum --check --status
tar -xzf "$FILEBROWSER_INSTALL_DIR/filebrowser.tar.gz" -C "$FILEBROWSER_INSTALL_DIR" filebrowser LICENSE
install -m 0755 "$FILEBROWSER_INSTALL_DIR/filebrowser" /usr/local/bin/filebrowser
install -D -m 0644 "$FILEBROWSER_INSTALL_DIR/LICENSE" /usr/local/share/licenses/filebrowser/LICENSE
/usr/local/bin/filebrowser version
