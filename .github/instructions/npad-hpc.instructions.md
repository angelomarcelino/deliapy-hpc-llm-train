---
applyTo: "**/*.{py,sh,slurm,md,toml,yml,yaml}"
description: "Diretrizes do cluster NPAD/UFRN, SLURM, armazenamento BeeGFS e integração DeLIApy/OpenMPI para este repositório."
---

# Diretrizes NPAD/HPC & DeLIApy

Sempre consulte [`AGENTS.md`](../../AGENTS.md), [`docs/npad_guide.md`](../../docs/npad_guide.md) e `AGENTS.local.md` (se presente localmente) antes de propor scripts de execução, diretórios de I/O ou alterações de ambiente.

## Checklist Obrigatório ao Gerar Código ou Scripts

1. **Separação Home vs. Scratch:**
   - Scripts e código em `~/deliapy-hpc-llm-train` (Home NFS, cota 100 GB soft / 1 TB hard).
   - Checkpoints do DeLIA (`ft_dir`), configs por job (`delia_params.json`, `delia_config.txt`, com `FT_FOLDER` terminando em `"/"`), datasets e caches (`PIP_CACHE_DIR`, `POETRY_CACHE_DIR`) sempre em `${SCRATCH_GLOBAL:-$HOME/scratch}` (BeeGFS `/mnt/beegfs`).
2. **Scripts SLURM (`.sh`):**
   - Linha 1: `#!/bin/bash`, seguida por todas as diretivas `#SBATCH` antes de comandos executáveis (use `#SBATCH --output=%x_%j.out` ou garanta `mkdir -p logs` antes do `sbatch`).
   - Partições de CPU disponíveis: `intel-128`, `intel-256`, `intel-512`, `amd-512`, `amd-3tb`.
   - Partições de GPU disponíveis (com `#SBATCH --gpus-per-node`): `gpu-8-v100`, `gpu-4-a100`, `gpu-8-h100`, `gpu-4-h200`. Na `gpu-4-a100`, o limite é 2000 MB de RAM/CPU (peça múltiplos CPUs com `--cpus-per-task`).
   - Ative o Conda (`eval "$(conda shell.bash hook)" && conda activate py311`) **antes** de `module load compilers/gnu/14.2.0 libraries/openmpi/5.0.10-gnu14-ucxmt`.
   - Lance jobs MPI (`mpi4py`/`deliapy`) com **`mpirun`** (sem `-np`), evitando conflitos de PMIx com `srun`.
3. **Build e Execução de `deliapy` e Python:**
   - Python `>=3.11,<3.12` (ambiente Conda `py311` com `--override-channels` e `libstdcxx-ng>=14` + `poetry config virtualenvs.create false`), tipagem estrita (sem `Any`) e logs/docstrings em Português (Brasil).
   - No NPAD, compile `deliapy` a partir do repositório Git (`--recurse-submodules`) com `MPI_HOME=/opt/npad/shared/libraries/openmpi/5.0.10-gnu14-ucxmt` (wheels/sdist do PyPI falham por `glibc 2.28`, RPATH `/opt/openmpi-5.0.8/lib` e falta de `.pyx`/`deliacpp`).
   - Pré-carregue `libmpi.so` de `MPI_HOME` com `ctypes.RTLD_GLOBAL` antes de importar `mpi4py`/`deliapy`; em CUDA, use `.detach().cpu().numpy()` ao reduzir gradientes via MPI.
4. **Citações na Documentação:**
   - Ao explicar fatos técnicos em documentação ou respostas, inclua links de referência inline (ex: Slurm docs, NPAD Tutorials, BeeGFS docs, Linux man pages).
