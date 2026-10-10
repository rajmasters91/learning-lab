### GIT

The mental model: your files move through three places.

Working directory: files you edit, like any other folder.
Staging area: files you've marked as "include this in the next snapshot" (git add).
Repository: the saved snapshots (git commit), which you then copy to GitHub (git push).

A commit is a named snapshot of the whole project at that moment. That's what lets you say "the Week 3 results were taken with this exact config."

One-time setup (about 15 minutes)

Create a GitHub account (if you don't have one), then create an empty repository named inference-lab.
On the node/VS code:
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git clone https://github.com/<you>/inference-lab.git
cd inference-lab

GitHub will ask you to authenticate on the first push. Follow its prompt; a personal access token is the simplest route.

Every time you change something

git status                      # what changed?
git add hardware/a100-80g.yaml  # stage it
git commit -m "Record A100 hardware profile for GPU 1"
git push

That's the whole Week 1 loop. Commit after each session, with a message that says what and why. Think "Lock GPU 1 clocks at 1410 MHz for repeatable benchmarks", not "update".

Two habits that pay off

Run git status before and after every commit until it becomes a reflex.
Never edit files in results/. Raw benchmark output is committed once and left alone.

Add a ## Git entry to notes/concepts.md once the three-places model clicks. Then carry on with Session 1.


One check: your HF_TOKEN goes in the docker run command. If you save that command to config.yaml, write HF_TOKEN=<redacted> rather than the real value.

what is .gitignore?

.gitignore is a plain text file at the root of the repo that lists files and patterns git should never track. Anything matching a line in it won't show up in git status and can't be committed by accident, even with git add ..

Create ~/learning-lab/.gitignore with:

# secrets
.env
*.token
secrets/

# large or regenerable
huggingface/
*.safetensors

# editor noise
.vscode/
.DS_Store

One pattern per line. A trailing / means "a directory", * is a wildcard, # starts a comment.

Commit the .gitignore itself; it's part of the repo. Then test it: create an empty .env file and run git status. If it doesn't appear, the ignore works.

Why it matters for you: in Session 2 your HF token is in a command line. If you ever save that to a file for convenience, call it .env, and git will refuse to see it. The *.safetensors line protects you from accidentally committing 16 GB of model weights, which GitHub would reject anyway but only after a long, confusing push.



## Reference

GPU Clock locking:
GPU's core clock (the SM clock) isn't fixed. It increases with workload and drops when the GPU reaches its power limit or gets hot. => same benchmark can run at 1,410 MHz one minute and 1,250 MHz the next, this affects benchmark numbers.

$ nvidia-smi -i 2 -q -d SUPPORTED_CLOCKS        ### lists the allowed values. For A100, SM clocks range from 210 MHz to 1410 MHz. 

Locking => telling the GPU to stay at one SM value: sudo nvidia-smi -i 2 -lgc 1410,1410.

Note: 1512 MHz memory clock on the A100 is fixed and cant be locked. 

Tradeoff: For A100, 1410 MHz is maximum. GPU under high load can reach 300W and hence drops GPU clock below 1410 MHz even when locked. A lower value (say 1200) holds steady but at reduced speed.

How to decide: start with 1410. During your first benchmark, watch nvidia-smi -i 2 -q -d CLOCK,PERFORMANCE and look at the actual SM clock and the throttle reasons. If it holds 1410, keep it. If it dips, lower the lock until it stays put.

Command Reference:
1. Diagnostics & Monitoring
⚬	nvidia-smi — View current GPU state, Persistence-M status, and active clocks.
⚬	sudo systemctl status nvidia-persistenced — Check if the persistence daemon is running and view its launch arguments.
⚬	sudo journalctl -eu nvidia-persistenced — View the startup logs and error history for the daemon.
⚬	sudo lsof /dev/nvid* — Verify if the daemon is actively holding the GPU character device files open.
⚬	nvidia-smi -q -d SUPPORTED_CLOCKS — List the maximum hardware-supported memory and graphics clocks for your GPUs.
2. Fixing the Persistence Daemon (Update-Resilient Drop-in)
⚬	sudo mkdir -p /etc/systemd/system/nvidia-persistenced.service.d/ — Create the systemd drop-in directory.
⚬	sudo bash -c 'echo -e "[Service]\nExecStart=\nExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced --verbose" > /etc/systemd/system/nvidia-persistenced.service.d/override.conf' — Create the override file to strip the --no-persistence-mode flag.
⚬	sudo systemctl daemon-reload — Tell systemd to re-read the configuration and detect the drop-in.
⚬	sudo systemctl restart nvidia-persistenced — Restart the daemon to apply the persistent state.
3. GPU Tuning & P0 Locking (Targeting Specific GPUs)
⚬	sudo nvidia-smi -i 2 -pm 1 — Manually toggle legacy persistence mode on for GPU 2 (if bypassing the daemon).
⚬	sudo nvidia-smi -i 2 -pl 250 — Set a strict 250W power limit for GPU 2.
⚬	sudo nvidia-smi -i 2 -lgc 1410,1410 — lock gpu clock to prevent thermal downclocking for GPU 2.
4. Making Automation Scripts Executable
⚬	sudo chmod +x /usr/local/bin/gpu-tune.sh — Grant execution permissions to your custom tuning bash script before hooking it into systemd.