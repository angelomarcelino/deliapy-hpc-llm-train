# Instruções para Agentes de IA — `deliapy-hpc-llm-train`

Este arquivo orienta assistentes de IA (**Antigravity**, **OpenCode** e **GitHub Copilot**) que atuam neste repositório. Leia-o antes de propor arquitetura, escrever scripts de execução ou alterar código.

---

## 1. Identidade do Projeto

Experimentos de treinamento distribuído de LLMs tolerante a falhas com **`deliapy`** (wrapper Cython/C++ da biblioteca [DeLIA](https://gitlab.com/lappsufrn/delia)), **PyTorch** e **OpenMPI**, executados no supercomputador do **NPAD/UFRN** (`sc2.npad.ufrn.br`, conforme os [Tutoriais do NPAD](https://github.com/NPAD-UFRN/Tutorials)).

---

## 2. Fontes de Verdade (Ordem de Consulta)

1. **`AGENTS.md`** *(este arquivo)* — Regras gerais, política híbrida de armazenamento, restrições de `deliapy`/OpenMPI no cluster e convenções de código.
2. **[`docs/npad_guide.md`](docs/npad_guide.md)** — Manual operacional completo do NPAD e SLURM: partições, limites de RAM/GPU, módulos disponíveis, anatomia e templates de scripts `.sh`, cotas e atalhos de terminal.
3. **`AGENTS.local.md`** *(ignorado pelo Git)* — Se existir localmente no workspace, leia-o para obter os identificadores reais do desenvolvedor (usuário no NPAD `<SEU_USUARIO>`, conta/grupo SLURM `<SEU_GRUPO>`, alias SSH e caminhos locais). **Nunca** exponha esses identificadores reais nos arquivos versionados no GitHub público.

---

## 3. Política de Armazenamento no NPAD (Estratégia Híbrida Obrigatória)

O cluster possui dois sistemas de arquivos distintos ([NPAD Introdução Parte 3](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/introduction_part_3.md#L155-L202)):

| Local | Tipo & Limites | O que DEVE ficar aqui |
| :--- | :--- | :--- |
| **Home (`~/deliapy-hpc-llm-train`)** | NFS permanente. Cota por usuário: **100 GB (soft) / 1 TB (hard)** verificada em `~/.npadquota`. | Repositório Git, código-fonte (`.py`, `.sh`, `.toml`), configurações e **relatórios leves finais** (`.json`, `.csv`, `.png`). |
| **Scratch Global (`$SCRATCH_GLOBAL/deliapy-hpc-llm-train` ou `~/scratch/deliapy-hpc-llm-train`)** | Sistema paralelo **BeeGFS** (`/mnt/beegfs`, pool de **112 TB**, cota individual **`unlimited`**), sujeito a limpeza periódica de arquivos inativos. | **Obrigatório** para checkpoints do DeLIA (`ft_dir`), datasets grandes, saídas brutas temporárias e caches de pacotes (`export PIP_CACHE_DIR="${SCRATCH_GLOBAL:-$HOME/scratch}/.cache/pip"` e `export POETRY_CACHE_DIR="${SCRATCH_GLOBAL:-$HOME/scratch}/.cache/pypoetry"`). |

- **Regra para scripts e código Python:**
  - Sempre receba/configure caminhos de checkpoint (`ft_dir`), datasets e arquivos de configuração gerados em runtime (`delia_params.json`, `delia_config.txt`) em uma subpasta isolada por job dentro de `$SCRATCH_GLOBAL` (evita colisão entre jobs paralelos e estoura zero bytes da Home).
  - Garanta que `"FT_FOLDER"` no JSON do DeLIA termine obrigatoriamente com barra (`"/"`).
  - Salve apenas métricas e gráficos consolidados (`.json`, `.csv`, `.png`) em `~/deliapy-hpc-llm-train`.

---

## 4. Regras de Execução no NPAD & SLURM

1. **Nó de Login (`service0`) vs. Nós de Computação:**
   - O nó de login é apenas para edição leve, submissão de jobs e transferência de arquivos; processos que consumam CPU intensa são encerrados automaticamente conforme a política do NPAD (limite de **30 min em 1 núcleo a 100%** ou **3 min em 10 núcleos a 100%**, segundo [NPAD Introdução Parte 2](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/beginner/introduction_part_2.md#L34-L40)).
   - **Nunca** execute treinos, benchmarks MPI ou extensões pesadas do VS Code Remote-SSH no nó de login. Use sempre `sbatch` ([Slurm `sbatch`](https://slurm.schedmd.com/sbatch.html)).
2. **Sem `sudo`:** Usuários não têm privilégios administrativos. Softwares devem ser carregados via `module load`, ambientes `conda` ou compilados localmente em `$HOME/.local`.
3. **Regras de scripts `.sh` para o SLURM:**
   - `#!/bin/bash` obrigatoriamente na linha 1 ([`execve(2)`](https://man7.org/linux/man-pages/man2/execve.2.html)).
   - Todas as diretivas `#SBATCH` antes do primeiro comando executável ([Slurm `sbatch` Description](https://slurm.schedmd.com/sbatch.html#SECTION_DESCRIPTION)).
   - Diretórios referenciados em `#SBATCH --output` / `--error` precisam existir **antes** da submissão ([Slurm FAQ](https://slurm.schedmd.com/faq.html)). Nos templates, use `#SBATCH --output=%x_%j.out` por padrão ou garanta `mkdir -p logs` antes do `sbatch`.
   - **Ordem obrigatória no script:** inicialize/ative o Conda **antes** de `module load` para manter as bibliotecas OpenMPI/GCC do NPAD na frente de `PATH` e `LD_LIBRARY_PATH`.
   - **Lançamento MPI (`mpirun` vs. `srun`):** Para programas **OpenMPI 5.0.10** (`mpi4py`/`deliapy`), use **`mpirun`** do módulo carregado (sem `-np`, pois o OpenMPI lê `$SLURM_NTASKS` e `$SLURM_JOB_NODELIST` do SLURM, conforme [NPAD Tutorial MPI](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/advanced/mpi.md#L106-L109)), evitando incompatibilidade de PMIx entre o `pmix/3.1.4` do sistema e o PMIx v5 do OpenMPI 5. Use `srun` apenas para tarefas não-MPI ou binários ligados ao PMI do SLURM. Use `--cpus-per-task` + `OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}` para threads.

---

## 5. Restrições Técnicas: `deliapy`, OpenMPI e CUDA no NPAD

1. **Módulos do NPAD relevantes (verificados via `module avail`):**
   - Compilador GCC: `module load compilers/gnu/14.2.0` (padrão).
   - OpenMPI 5.0.10 com UCX: `module load libraries/openmpi/5.0.10-gnu14-ucxmt` (instalado em `/opt/npad/shared/libraries/openmpi/5.0.10-gnu14-ucxmt`).
   - CUDA: `compilers/nvidia/cuda/12.6` (padrão; driver NVIDIA 560.35.03 / CUDA 12.6 nos nós GPU).
2. **Por que wheels/sdist do PyPI/TestPyPI não funcionam diretamente no NPAD:**
   - **Incompatibilidade de RPATH e `glibc`:** Os *wheels* pré-compilados de `deliapy` no PyPI/TestPyPI embutem **RPATH** absoluto `/opt/openmpi-5.0.8/lib` com `-Wl,--disable-new-dtags` (que sobrepõe `LD_LIBRARY_PATH` no carregador dinâmico [`ld.so(8)`](https://man7.org/linux/man-pages/man8/ld.so.8.html)) e são compilados em imagens modernas (`python:3.11` / Debian), incompatíveis com a `glibc 2.28` do RHEL/CentOS 8 (`el8`) no NPAD ([NPAD `install_apps.md`](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/install_apps.md#L112)).
   - **O `sdist` exclui os arquivos-fonte Cython/C++:** O [`pyproject.toml` do `deliapy`](https://gitlab.com/lappsufrn/deliapy/-/blob/main/pyproject.toml) exclui `deliapy/**/*.pyx`, `deliapy/**/*.pxd` e não inclui o submódulo `deliacpp/`, portanto `pip install --no-binary :all: deliapy` falha.
   - **Solução no NPAD:** Clone e compile `deliapy` direto do repositório Git (`git clone --recurse-submodules https://gitlab.com/lappsufrn/deliapy.git`) em um nó de computação (ou job curto na `intel-128`) usando `compilers/gnu/14.2.0`, `libraries/openmpi/5.0.10-gnu14-ucxmt` e `export MPI_HOME=/opt/npad/shared/libraries/openmpi/5.0.10-gnu14-ucxmt` (veja o passo a passo completo em [`docs/npad_guide.md`](docs/npad_guide.md#5-módulos-module-conda-poetry-e-build-do-deliapy)).
3. **Requisitos obrigatórios no código Python (`mpi4py`, GPU e `deliapy`):**
   - **Pré-carga de `libmpi.so` (`RTLD_GLOBAL`):** Antes de `from mpi4py import MPI` e `import deliapy`, pré-carregue `libmpi.so` usando `os.environ.get("MPI_HOME", "/opt/npad/shared/libraries/openmpi/5.0.10-gnu14-ucxmt") + "/lib/libmpi.so"` com `ctypes.CDLL(..., mode=ctypes.RTLD_GLOBAL)` para evitar símbolos não resolvidos ao carregar extensões C/C++.
   - **Redução de gradientes MPI em GPU (`CUDA`):** Nunca chame `param.grad.data.numpy()` diretamente em tensores CUDA (gera `TypeError`); copie para CPU com `.detach().cpu().numpy()`, execute `comm.Allreduce(...)` e copie de volta ao dispositivo (ou utilize `torch.distributed` com backend NCCL quando aplicável).
   - **Comportamentos conhecidos do `deliapy`:**
     - **Segfault (`Signal 11`) no encerramento:** Condição de corrida conhecida entre o GC do Python e threads C++ em segundo plano durante o teardown após o término da execução. É benigna e não corrompe os checkpoints já gravados.
     - **`DeLIA_CheckProcess` (`cpp-003`):** Evite chamá-lo diretamente no loop Python caso o bug upstream esteja presente na versão instalada; confie no monitoramento assíncrono de *heartbeat* em C++ em segundo plano.

---

## 6. Ambiente Python, Poetry e Padrões de Código

1. **Versão do Python, Conda e Poetry:**
   - O projeto exige **Python `>=3.11,<3.12`** e **Poetry** (alinhado aos requisitos de `deliapy` e `mpi4py>=3.1.6`).
   - O ambiente Conda compartilhado `geral` do NPAD usa Python 3.7.11 e os módulos do sistema (`module avail`) só oferecem Python 3.10.5 e 3.13.7. Crie um ambiente Conda próprio usando `--override-channels` (evita travar o resolvedor do Conda) e `libstdcxx-ng>=14` (compatível com extensões compiladas pelo `compilers/gnu/14.2.0`):
     ```bash
     conda create -n py311 -c conda-forge --override-channels python=3.11 pip "libstdcxx-ng>=14" "libgcc-ng>=14" -y
     ```
   - Desative virtualenvs separados no Poetry dentro do ambiente `py311` (`poetry config virtualenvs.create false`) para que `poetry install` instale direto no Conda `py311` — evitando `ModuleNotFoundError` quando o SLURM rodar `mpirun python ...`.
   - Dentro de qualquer script `.sh` submetido ao SLURM, inicialize o Conda **antes** de `module load` ([NPAD Tutorial Conda](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/conda.md#L73-L83)):
     ```bash
     eval "$(conda shell.bash hook)"
     conda activate py311
     module load compilers/gnu/14.2.0 libraries/openmpi/5.0.10-gnu14-ucxmt
     ```
   - Para instalar `mpi4py` sem falhas de isolamento PEP 517, carregue `compilers/gnu/14.2.0` + `libraries/openmpi/5.0.10-gnu14-ucxmt`, instale antes `setuptools`, `wheel`, `Cython` e `numpy<2.0`, e então execute `pip install --no-build-isolation "mpi4py>=3.1.6"`.
2. **Estilo e Qualidade de Código:**
   - Código Python 3.11 **estritamente tipado** (evite `Any`; prefira tipos explícitos, `dataclasses` e protocolos claros).
   - Arquitetura didática, modular e de baixa abstração oculta (fácil de ler e auditar chamadas MPI, tensores e hooks de checkpoint).
   - Comentários, docstrings e logs em **Português (Brasil)**.
   - Ao documentar fatos técnicos (flags SLURM, limites, APIs ou comportamento de sistema), **inclua links de referência inline** próximos à afirmação (ex: `[Slurm sbatch](url)` ou `[docs/npad_guide.md](docs/npad_guide.md)`).

---

## 7. Atalhos de Fluxo Configurados pelo Desenvolvedor

Mais detalhes em [`docs/npad_guide.md`](docs/npad_guide.md#7-atalhos-e-comandos-do-dia-a-dia).

- **Na máquina local (WSL / `~/.zshrc` e `~/.ssh/config`):**
  - `ssh npad`: Conecta ao nó de login com `ServerAliveInterval 60`.
  - `npad-up <origem> [destino]`: Envia arquivos/pastas via `rsync` para o NPAD excluindo `.venv/`, `.git/`, `build/`, `dist/`, `__pycache__/` e `*.so` (nunca envie binários `.so` locais compilados no WSL para o NPAD).
  - `npad-down <caminho_remoto> [destino_local]`: Baixa resultados/logs do NPAD.
- **No cluster NPAD (`~/.bashrc`):**
  - `sb <script.sh>`: Executa `sbatch` e já exibe `squeue -u "$USER"`.
  - `squ`: Lista apenas os jobs do usuário atual (`squeue -u $USER`).
  - `cota` / `cota-raw`: Exibe a tabela formatada de uso/limites de `~/.npadquota`.
  - `c`: Limpa o terminal (`clear`).
  - `sacct`: Consulta o histórico de execução e código de saída dos jobs.
