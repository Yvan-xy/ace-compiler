#!/usr/bin/env bash
set -euo pipefail

EXPECTED_BASE="docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749"
EXPECTED_CONFIG="sha256:0131784115794405cb36a8068a82d7aea0937196d1d6e844b9dd021252ccf7e4"
OPENSSH_VERSION="1:8.9p1-3ubuntu0.16"

if [[ "${ACE_RUNPOD_BASE_IMAGE:-}" != "${EXPECTED_BASE}" ||
      "${ACE_RUNPOD_BASE_CONFIG_DIGEST:-}" != "${EXPECTED_CONFIG}" ]]; then
  echo "RunPod startup base identity mismatch" >&2
  exit 1
fi
printf '%s\n' \
  '1bf0e470c9bea818ddf7c73e83a06a70c8e3f1d2b03b51392ab2966829d5b00a  /etc/os-release' \
  '6be39758fb262923ecb7b991082c4735957b9beb912c2018f281918be749ccf0  /usr/local/cuda/include/cuda_runtime.h' \
  '2c17a51b5c6c248e92f71284d5c449240c3d649f2e6738ae2238d184ffe4013b  /usr/local/cuda/lib64/libcudadevrt.a' \
  'e4196076c5496c4bb5509be61e3d1cddf36b92a449a10ece1779afce3c65e684  /NGC-DL-CONTAINER-LICENSE' |
  sha256sum -c -
if [[ ! "${ACE_RUNPOD_SSH_PUBLIC_KEY:-}" =~ ^ssh-ed25519\ [A-Za-z0-9+/=]+\ [A-Za-z0-9._@-]+$ ]]; then
  echo "RunPod startup received an invalid one-use SSH public key" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
if [[ "$(apt-cache policy openssh-server | awk '/Candidate:/ {print $2}')" != "${OPENSSH_VERSION}" ]]; then
  echo "pinned openssh-server version is unavailable" >&2
  exit 1
fi
apt-get install -y --no-install-recommends "openssh-server=${OPENSSH_VERSION}"

install -d -m 0700 /root/.ssh
printf 'restrict %s\n' "${ACE_RUNPOD_SSH_PUBLIC_KEY}" >/root/.ssh/authorized_keys
chmod 0600 /root/.ssh/authorized_keys
rm -f /etc/ssh/ssh_host_*
ssh-keygen -q -t ed25519 -N '' -f /etc/ssh/ssh_host_ed25519_key
install -d -m 0755 /run/sshd
cat >/etc/ssh/sshd_config.d/ace-runpod.conf <<'EOF'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin prohibit-password
PubkeyAuthentication yes
AllowAgentForwarding no
AllowTcpForwarding no
X11Forwarding no
PermitTunnel no
GatewayPorts no
EOF
/usr/sbin/sshd -t
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
exec /usr/sbin/sshd -D -e
