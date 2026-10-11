### GIT

Git mental model: files moves via 3 places.

1. Working directory: user edit
2. Staging area: after `git add` i.e., files to be included in next snapshot.
3. Repository: saved snapshots (git commit), which is then copied to GitHub (git push).

On the node/VS code:
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git clone https://github.com/rajmasters91/learning-lab.git
cd inference-lab

After any edit: 
git status                      # what changed?
git add hardware/a100-80g.yaml  # stage it
git commit -m "Record A100 hardware profile for GPU 1"
git push


.gitignore?
plain text file at the root of the repo that lists files and patterns git should never track. Anything matching a line in it won't show up in git status and can't be committed by accident, even with git add.

Create ~/learning-lab/.gitignore with:

# secrets
.env
*.token
secrets/

# large or regenerable
huggingface/
*.safetensors #protects from accidentally committing 16GB of model weights

# editor noise
.vscode/
.DS_Store



## GPU

journalctl -eu nvidia-persistenced #startup logs and error history of daemon.

lsof /dev/nvid* — Verify if the daemon is actively holding the GPU character device files open.

bash scripts must have execution permission if it needs to added in a systemd services. Ex: chmod +x /usr/local/bin/gpu-tune.sh 