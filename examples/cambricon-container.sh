#!/bin/bash
# Run a command (default: a login shell) inside the Cambricon PyTorch container, with the MLU
# cards, this checkout and the work disk visible at their host paths.
#
#   bash examples/cambricon-container.sh                       # interactive shell
#   PREFLIGHT_ONLY=1 bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads din 0
#   SMOKE=1 bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads din 0
#   bash examples/cambricon-container.sh --stop                # remove the container
#
# Why a container: the MLU590 host has Python 3.8 and no NEUWARE, while FlagScale needs >= 3.10.
# The 26.08 image carries Python 3.12, torch 2.12.1 and torch_mlu 1.34.1 (examples/CAMBRICON.md).
#
# One long-lived container per user: the first call loads the image if needed and starts it
# (`sleep infinity`), later calls `docker exec` into it, so a training run does not depend on a
# container being recreated.  Mounts are fixed when the container is created; after changing
# WORK_ROOT or EXTRA_MOUNTS, run --stop first.
#
# Devices: every character device under /dev/cambricon* on the host is passed through, so no
# device name is hard-coded.  If the preflight inside then sees no card, retry once with
# PRIVILEGED=1 (after --stop) and record which node was missing in CAMBRICON.md.
#
# Docker needs sudo on this host (the login user is not in the docker group).  Processes in the
# container run as root, so files they write under WORK_ROOT are root-owned.
#
# Environment:
#   CAMBRICON_IMAGE_TAR  image tarball from the vendor bundle, used when the image is not loaded
#                        TODO(cluster): set the default to its path on shared storage
#   CAMBRICON_IMAGE      image reference, if already loaded (printed after the first load)
#   CONTAINER_NAME       default adrec-cambricon-$USER
#   WORK_ROOT            host directory for data and outputs, mounted at the same path
#                        (default /srv/$USER: `/` has about 100 GB free, /srv about 6 TB)
#   EXTRA_MOUNTS         more host paths to mount at the same path, separated by spaces;
#                        /flagos-ads and /flagos-rec are added when they exist
#   PRIVILEGED=1         use --privileged instead of per-device passthrough
#   DOCKER               docker command, default "sudo docker"
#
# Forwarded into the container when set: PREFLIGHT_ONLY SMOKE ALLOW_CPU_FALLBACK ACTION
# ADREC_VENV CAMBRICON_ENV_SH ADREC_DATA_ROOT ADREC_OUT_ROOT FLAGSCALE_PIN_MEMORY
# OMP_NUM_THREADS WANDB_MODE.  ADREC_OUT_ROOT defaults to $WORK_ROOT/flagos-adrec-cambricon.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
read -r -a docker <<< "${DOCKER:-sudo docker}"
name=${CONTAINER_NAME:-adrec-cambricon-${USER:-user}}

if [ "${1:-}" = "--stop" ]; then
  "${docker[@]}" rm -f "$name"
  exit $?
fi

WORK_ROOT=${WORK_ROOT:-/srv/${USER:-user}}
if ! mkdir -p "$WORK_ROOT" 2>/dev/null; then
  echo "cannot create WORK_ROOT=$WORK_ROOT; set WORK_ROOT to a writable directory on /srv" >&2
  exit 2
fi
export ADREC_OUT_ROOT=${ADREC_OUT_ROOT:-$WORK_ROOT/flagos-adrec-cambricon}

state=$("${docker[@]}" container inspect -f '{{.State.Running}}' "$name" 2>/dev/null)
if [ -z "$state" ]; then
  # ---- image -----------------------------------------------------------------------------
  image=${CAMBRICON_IMAGE:-}
  if [ -n "$image" ] && ! "${docker[@]}" image inspect "$image" >/dev/null 2>&1; then
    echo "image $image is not loaded; loading it from CAMBRICON_IMAGE_TAR" >&2
    image=""
  fi
  if [ -z "$image" ]; then
    tar=${CAMBRICON_IMAGE_TAR:-}
    if [ -z "$tar" ] || [ ! -f "$tar" ]; then
      echo "set CAMBRICON_IMAGE (already loaded) or CAMBRICON_IMAGE_TAR (the .tar.gz from the" >&2
      echo "vendor bundle, e.g. cambricon_pytorch_container-v26.08.0-torch2.12.1-...-py312.tar.gz)" >&2
      exit 2
    fi
    echo "===== loading $tar (takes a few minutes) ====="
    loaded=$("${docker[@]}" load -i "$tar") || exit 1
    echo "$loaded"
    image=$(sed -n 's/^Loaded image: //p; s/^Loaded image ID: //p' <<< "$loaded" | tail -1)
    if [ -z "$image" ]; then echo "could not read the image name from docker load" >&2; exit 1; fi
    echo "loaded $image; next time: export CAMBRICON_IMAGE=$image"
  fi

  # ---- devices ---------------------------------------------------------------------------
  device_args=()
  if [ "${PRIVILEGED:-0}" = 1 ]; then
    device_args=(--privileged)
  else
    shopt -s nullglob
    for dev in /dev/cambricon* /dev/cambricon*/*; do
      if [ -c "$dev" ]; then device_args+=(--device "$dev:$dev"); fi
    done
    shopt -u nullglob
    if [ ${#device_args[@]} -eq 0 ]; then
      echo "no /dev/cambricon* character devices on this host; is the driver loaded (cnmon)?" >&2
      exit 1
    fi
  fi

  # ---- mounts ----------------------------------------------------------------------------
  mount_args=(-v "$ROOT:$ROOT" -v "$WORK_ROOT:$WORK_ROOT")
  for path in /flagos-ads /flagos-rec ${EXTRA_MOUNTS:-}; do
    if [ -e "$path" ]; then mount_args+=(-v "$path:$path"); fi
  done
  # The host's cnmon matches the host driver; the image's copy may not.
  if [ -x /usr/bin/cnmon ]; then mount_args+=(-v /usr/bin/cnmon:/usr/bin/cnmon:ro); fi

  echo "===== starting $name from $image ====="
  if [ "${PRIVILEGED:-0}" = 1 ]; then
    echo "devices: all (--privileged)"
  else
    echo "devices: $((${#device_args[@]} / 2)) /dev/cambricon* nodes passed through"
  fi
  echo "mounts : ${mount_args[*]}"
  "${docker[@]}" run -d --name "$name" --init \
    --network host --ipc host \
    "${device_args[@]}" "${mount_args[@]}" \
    -w "$ROOT" --entrypoint sleep "$image" infinity >/dev/null || exit 1
elif [ "$state" != "true" ]; then
  "${docker[@]}" start "$name" >/dev/null || exit 1
fi

# ---- exec --------------------------------------------------------------------------------
env_args=()
for var in PREFLIGHT_ONLY SMOKE ALLOW_CPU_FALLBACK ACTION ADREC_VENV CAMBRICON_ENV_SH \
           ADREC_DATA_ROOT ADREC_OUT_ROOT FLAGSCALE_PIN_MEMORY OMP_NUM_THREADS WANDB_MODE; do
  if [ -n "${!var+x}" ]; then env_args+=(-e "$var=${!var}"); fi
done
tty_args=(-i)
if [ -t 0 ] && [ -t 1 ]; then tty_args=(-it); fi
if [ $# -eq 0 ]; then set -- bash; fi
# A login shell picks up /etc/profile.d, where an image may put its python environment.
exec "${docker[@]}" exec "${tty_args[@]}" "${env_args[@]}" -w "$ROOT" "$name" \
  bash -lc 'cd "$0" && exec "$@"' "$ROOT" "$@"
