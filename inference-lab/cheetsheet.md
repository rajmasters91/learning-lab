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



