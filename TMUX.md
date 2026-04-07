# 7-Day Interactive Slurm Session (Survives Cursor/SSH Reconnects)

This guide is tailored for user `s4374886` and the A100-80G partition.

## Why this is needed

If you run `srun --pty ...` directly in a normal SSH shell, reconnects can drop the controlling TTY and kill your interactive step.

Running inside `tmux` keeps the session alive across disconnects.

---

## 0) One-time: check your account/QoS limits

Some clusters require `--account`, and many enforce max walltime per partition/QoS.

```bash
# Show available associations for your user (if allowed)
sacctmgr -n show assoc where user=$USER format=account,partition,qos%30

# If your site has this command:
sacctmgr -n show user $USER withassoc format=user,account,partition,qos
```

If your admin requires account flag, note your account name and use:
`--account=<YOUR_ACCOUNT>`

---

## 1) SSH to login node

```bash
ssh alice
```

---

## 2) Start a persistent tmux on login node

```bash
tmux new -s slurm7d
```

Detach anytime with: `Ctrl-b` then `d`.

---

## 3) Request your interactive GPU session (inside tmux)

Use your command, optionally adding `--account=...` if your site requires it:

```bash
srun \
  --partition=gpu-a100-80g \
  --gres=gpu:a100:1 \
  --cpus-per-task=8 \
  --mem=120g \
  --time=7-00:00:00 \
  --pty bash
```

If account is required:

```bash
srun \
  --account=<YOUR_ACCOUNT> \
  --partition=gpu-a100-80g \
  --gres=gpu:a100:1 \
  --cpus-per-task=8 \
  --mem=120g \
  --time=7-00:00:00 \
  --pty bash
```

---

## 4) (Recommended) Start nested tmux on compute node

Once `srun --pty bash` opens on compute node:

```bash
tmux new -s work
```

Run all training/interactive commands inside this nested `tmux`.

This gives two protection layers:
- login node tmux (`slurm7d`)
- compute node tmux (`work`)

---

## 5) Safely disconnect / reconnect

### Disconnect safely
- In compute-node tmux: `Ctrl-b d`
- In login-node tmux: `Ctrl-b d`

Then you can close SSH or Cursor can reconnect; job stays alive.

### Reconnect
```bash
ssh alice
tmux attach -t slurm7d
```

If nested tmux exists on compute node:
```bash
tmux attach -t work
```

---

## 6) Monitor your job

```bash
squeue -u $USER
```

Detailed:
```bash
squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.10l %.6D %R"
```

---

## 7) End the session cleanly

From login node or any node with Slurm access:

```bash
scancel <jobid>
```

Or just exit shells and stop tmux sessions when done.

---

## Notes / gotchas

- 7 days only works if partition/QoS allows `7-00:00:00`.
- If max walltime is lower, Slurm will terminate at that limit.
- `tmux` prevents SSH reconnect from killing your interactive session.
- `nohup` is not a substitute for interactive `srun --pty`.

---

## Multiple Bash Sessions in tmux (Like Multiple SSH Terminals)

Yes, you can get the same multi-shell workflow after `srun --pty bash`, but inside tmux.

### A) Use tmux windows (recommended)

After `srun --pty bash` lands you on the compute node:

```bash
tmux new -s work
```

Then:
- New window (new shell): `Ctrl-b c`
- Next/prev window: `Ctrl-b n` / `Ctrl-b p`
- Window picker: `Ctrl-b w`
- Rename current window: `Ctrl-b ,`

Each window is an independent shell, similar to opening another SSH terminal.

### B) Use tmux panes (split-screen shells)

Inside a tmux window:
- Vertical split: `Ctrl-b %`
- Horizontal split: `Ctrl-b "`
- Move across panes: `Ctrl-b` + arrow key
- Close a pane: `exit` in that pane

### C) Use multiple named tmux sessions

If you prefer separate workspaces:

```bash
tmux new -s train
tmux new -s monitor
tmux new -s debug
```

List and attach:

```bash
tmux ls
tmux attach -t train
```

### D) Reconnect flow

After reconnect:

```bash
ssh <your-cluster-login>
tmux attach -t slurm7d
```

Then on the compute node:

```bash
tmux attach -t work
```

Recommended structure:
- Login node tmux: `slurm7d` (keeps allocation chain alive)
- Compute node tmux: `work` (where you keep multiple shells/windows/panes)
