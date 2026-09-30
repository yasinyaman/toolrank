#!/usr/bin/env bash
# `as-toolrank COMMAND [ARGS...]` in the toolrank-vllm image: run a command as the unprivileged
# user toolrank runs as. /usr/local/bin/toolrank is `as-toolrank /opt/toolrank/venv/bin/toolrank`,
# so toolrank and the stdio MCP servers it starts (npx, uvx: other people's code) are never root
# here, whether the entrypoint starts them or `docker exec` does; vLLM keeps the container's user
# (root: the GPU and /models).
#
# Who that user is, in a container that runs as root, follows the directory the command writes to:
# its --out; for serve and search their --data (cache, index and usage log live there); else the
# image's /data. A path that does not exist yet counts as its nearest existing parent.
#   - Its owner, when that is not root: a bind-mounted directory stays its owner's, and a file only
#     that owner can read (a 0600 config next to it) stays readable. The group comes along unless
#     it is root's.
#   - Else the image's `toolrank` user.
# Root's files on the /data volume (a directory Docker created for a bind mount, what a root
# `docker exec` wrote) are handed to that user, there and nowhere else. Nobody else's files are
# re-owned, symlinks are not followed, other mounts are left alone, and a chown that fails is a
# warning, not the end of the container.
#
# Under a rootless engine uid 0 is the user who runs it, and a directory that looks like root's is
# theirs: nothing is dropped and nothing re-owned, since any other uid would lock them out of it. A
# directory owned by a user the container cannot map (root's, seen from there) gets the image's user
# and a warning.
# A container started as another user (`--user`) drops nothing either; the command gets a home it
# can write if it has none. In every case uv loses what vLLM's image sets for root and for vLLM's
# own environment (the cache directory, an override, the index strategy): its cache goes under the
# home.
#
#   TOOLRANK_ENTRYPOINT_DRY_RUN=1    print the command line and exit
set -euo pipefail

dry=${TOOLRANK_ENTRYPOINT_DRY_RUN:-}
cmd=(env -u UV_CACHE_DIR -u UV_OVERRIDE -u UV_INDEX_STRATEGY)
warn() { echo "as-toolrank: $*" >&2; }

# uid 0 in this container is not the host's root: a rootless engine (or a user namespace)
mapped_root() {
  local inside outside rest
  while read -r inside outside rest; do
    [[ "$inside" == 0 && "$outside" != 0 ]] && return 0
  done <<<"$(cat /proc/self/uid_map 2>/dev/null || true)"
  return 1
}

if [[ "$(id -u)" == 0 ]]; then
  out='' data='' in_data='' prev=''
  for arg in "$@"; do
    case "$prev" in --out) out=$arg ;; --data) data=$arg ;; esac
    case "$arg" in
      --out=*) out=${arg#*=} ;;
      --data=*) data=${arg#*=} ;;
      serve | search) in_data=1 ;; # they write where they read; eval and finetune only read --data
    esac
    prev=$arg
  done
  dir=$out
  [[ -n "$dir" || -z "$in_data" ]] || dir=$data
  dir=${dir:-/data}
  while [[ ! -d "$dir" ]]; do dir=$(dirname "$dir"); done
  read -r owner mount <<<"$(stat -c '%u:%g %m' "$dir" 2>/dev/null || echo '0:0 /')"
  uid=${owner%%:*}
  gid=${owner##*:}
  image_uid=$(id -u toolrank)
  image_gid=$(id -g toolrank)
  nobody=$(cat /proc/sys/kernel/overflowuid 2>/dev/null || echo 65534)
  if [[ "$uid" == 0 ]] && mapped_root; then
    uid='' # the engine's own user: as it is
  else
    if [[ "$uid" == "$nobody" ]]; then # an owner this container cannot map (a rootless engine, someone else's directory)
      warn "$dir belongs to a user this container cannot map: running as the image's user, which may not be able to write it"
      uid=0
      gid=0
    fi
    [[ "$uid" != 0 ]] || uid=$image_uid
    [[ "$gid" != 0 ]] || gid=$image_gid # toolrank is never in the root group
    # under /home, which only root can change: the user cannot swap its home for a link
    home=/home/toolrank
    [[ "$uid" == "$image_uid" ]] || home=/home/toolrank-$uid
    if [[ -z "$dry" ]]; then
      [[ -d "$home" ]] || install -d -m 0750 -o "$uid" -g "$gid" "$home" 2>/dev/null ||
        warn "no home for uid $uid ($home cannot be made): npx and uvx have nowhere to keep a cache"
      if [[ "$mount" == /data ]] && ! find /data -xdev -user root -exec chown -h "$uid:$gid" {} + 2>/dev/null; then
        warn "some of root's files under /data could not be handed to uid $uid; toolrank may not be able to write them"
      fi
    fi
    cmd+=(HOME="$home" setpriv --reuid="$uid" --regid="$gid" --clear-groups --no-new-privs)
  fi
elif [[ ! -w "${HOME:-/}" ]]; then
  # no home this user can write (a uid without a passwd entry): a private one, made for this run
  home=${TMPDIR:-/tmp}/toolrank-home.XXXXXX
  [[ -n "$dry" ]] || home=$(mktemp -d "$home" 2>/dev/null) || home=''
  [[ -z "$home" ]] || cmd+=(HOME="$home")
fi

if [[ -n "$dry" ]]; then
  echo "${cmd[*]} $*"
  exit 0
fi
exec "${cmd[@]}" "$@"
