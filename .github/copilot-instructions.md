# Instruções do Repositório para o GitHub Copilot

Este repositório contém experimentos de treinamento distribuído de LLMs tolerante a falhas usando **`deliapy`**, **PyTorch** e **OpenMPI** no supercomputador do **NPAD/UFRN** (`sc2.npad.ufrn.br`).

## Fontes de Verdade

Consulte sempre os documentos abaixo antes de escrever scripts SLURM, configurar ambientes ou alterar código Python:
- [`AGENTS.md`](../AGENTS.md): Regras gerais, estratégia híbrida de armazenamento, restrições de `deliapy`/OpenMPI e padrões de código.
- [`docs/npad_guide.md`](../docs/npad_guide.md): Manual completo do NPAD (partições, limites de RAM/GPU, módulos, cotas, templates `.sh` e atalhos).
- `AGENTS.local.md` *(se existir no workspace local, ignorado pelo Git)*: Identificadores reais do ambiente do desenvolvedor (usuário, conta/grupo SLURM e caminhos locais). Nunca exponha dados reais de `AGENTS.local.md` em arquivos versionados.

## Regras Essenciais

1. **Armazenamento Híbrido no NPAD:**
   - **Home (`~/deliapy-hpc-llm-train`):** Repositório Git, código-fonte e relatórios leves (`.json`, `.png`). Possui cota de 100 GB (soft) / 1 TB (hard) em `~/.npadquota`.
   - **Scratch (`${SCRATCH_GLOBAL:-$HOME/scratch}/deliapy-hpc-llm-train`):** Obrigatório para checkpoints do DeLIA (`ft_dir`), `delia_params.json`/`delia_config.txt` isolados por job (com `FT_FOLDER` terminando em `"/"`), datasets grandes e caches (`PIP_CACHE_DIR`, `POETRY_CACHE_DIR`). Usa BeeGFS (pool de 112 TB, cota individual `unlimited`, sujeito a limpeza de arquivos inativos).
2. **Regras de Execução SLURM (`.sh`):**
   - Nunca execute cargas pesadas no nó de login (`service0`); submeta via `sbatch`.
   - Mantenha `#!/bin/bash` na linha 1 e todas as diretivas `#SBATCH` antes do primeiro comando executável.
   - Prefira `#SBATCH --output=%x_%j.out` ou garanta que diretórios de `--output` / `--error` (ex: `logs/`) existam **antes** do `sbatch`.
   - Nos scripts `.sh`, ative o Conda **antes** de carregar os módulos (`eval "$(conda shell.bash hook)" && conda activate py311` antes de `module load compilers/gnu/14.2.0 libraries/openmpi/5.0.10-gnu14-ucxmt`).
   - Use **`mpirun`** (sem `-np`) do OpenMPI 5.0.10 para lançar processos MPI (`mpi4py`/`deliapy`), evitando conflitos de PMIx com `srun`, e configure `OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}` para threads.
3. **Restrições de `deliapy`, OpenMPI e CUDA:**
   - Wheels/sdist de `deliapy` no PyPI/TestPyPI não funcionam diretamente no NPAD (`glibc 2.28` no RHEL/CentOS 8, RPATH absoluto `/opt/openmpi-5.0.8/lib` com `-Wl,--disable-new-dtags`, e `sdist` sem `.pyx`/`.pxd`/`deliacpp/`). Compile o `deliapy` no NPAD a partir de `git clone --recurse-submodules https://gitlab.com/lappsufrn/deliapy.git` com `export MPI_HOME=/opt/npad/shared/libraries/openmpi/5.0.10-gnu14-ucxmt` (após instalar `setuptools`, `wheel`, `Cython`, `numpy<2.0` e `mpi4py` com `--no-build-isolation`).
   - No código Python, pré-carregue `libmpi.so` via `ctypes.CDLL(os.path.join(os.environ.get("MPI_HOME", "/opt/npad/shared/libraries/openmpi/5.0.10-gnu14-ucxmt"), "lib/libmpi.so"), mode=ctypes.RTLD_GLOBAL)` antes de importar `mpi4py`/`deliapy`, e use `.detach().cpu().numpy()` ao reduzir gradientes CUDA via MPI.
   - Considere benigno um eventual `Signal 11` no teardown após conclusão de testes/execuções MPI; evite `DeLIA_CheckProcess` no loop Python (`cpp-003`) e confie no monitoramento por *heartbeat* C++ em background.
4. **Python, Estilo e Referências:**
   - Crie o ambiente próprio com `conda create -n py311 -c conda-forge --override-channels python=3.11 pip "libstdcxx-ng>=14" "libgcc-ng>=14" -y` e configure `poetry config virtualenvs.create false` para instalar direto no `py311`.
   - Use tipagem estrita (sem `Any`) e comentários/docstrings/logs em Português (Brasil).
   - Ao afirmar detalhes técnicos (flags SLURM, limites de partições, comandos Linux ou comportamento de APIs), inclua links de referência inline próximos ao texto.
