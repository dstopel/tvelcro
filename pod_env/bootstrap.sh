#!/bin/bash
# per-pod bootstrap: fast, idempotent; container fs is ephemeral, volume is not
set -u
M="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # /workspace/env on the volume, repo checkout otherwise
command -v zsh >/dev/null || { apt-get update -qq && apt-get install -y -qq zsh vim tmux htop ripgrep >/dev/null 2>&1; }
[ -d $M/ohmyzsh ] || git clone -q --depth 1 https://github.com/ohmyzsh/ohmyzsh $M/ohmyzsh
ln -sf $M/zshrc /root/.zshrc
ln -sf $M/vimrc /root/.vimrc
ln -sf $M/tmux.conf /root/.tmux.conf
[ -d /workspace ] && { grep -q env/activate /root/.bashrc 2>/dev/null || echo "source $M/activate.sh" >> /root/.bashrc; }  # activate.sh is volume-specific
chsh -s "$(command -v zsh)" root 2>/dev/null || true
echo "bootstrap done"
