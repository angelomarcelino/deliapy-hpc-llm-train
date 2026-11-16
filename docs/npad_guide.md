# Guia Operacional do NPAD/UFRN e SLURM

Referência técnica consolidada para conexão, armazenamento, submissão de jobs via SLURM e execução de experimentos com Python, PyTorch, OpenMPI e `deliapy` no cluster do **NPAD (Núcleo de Processamento de Alto Desempenho — UFRN)**.

---

## 1. Acesso e Conexão SSH

O acesso ao supercomputador ocorre via SSH na porta `4422` do servidor `sc2.npad.ufrn.br` ([NPAD Introdução Parte 1](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/beginner/introduction_part_1.md#L110-L140)).

### Configuração no `~/.ssh/config` (máquina local)

Adicione o bloco abaixo ao `~/.ssh/config` da sua máquina local (com permissão `chmod 600 ~/.ssh/config`) para simplificar o login para `ssh npad`, manter a sessão viva sem congelar por inatividade ([`ssh_config(5)`](https://man7.org/linux/man-pages/man5/ssh_config.5.html)) e registrar a chave no agente automaticamente:

```text
Host npad
    HostName sc2.npad.ufrn.br
    Port 4422
    User <SEU_USUARIO>
    IdentityFile ~/.ssh/id_rsa_npad
    AddKeysToAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

### Compartilhamento opcional de `ssh-agent` no `~/.zshrc`

Para digitar a *passphrase* da chave apenas uma vez por inicialização do sistema em vez de em cada aba/chamada de `ssh` ou `rsync` ([`ssh-agent(1)`](https://man7.org/linux/man-pages/man1/ssh-agent.1.html)):

```zsh
export SSH_AUTH_SOCK="$HOME/.ssh/ssh-agent.sock"
if ! ssh-add -l >/dev/null 2>&1; then
    rm -f "$SSH_AUTH_SOCK"
    eval "$(ssh-agent -a "$SSH_AUTH_SOCK" -s)" >/dev/null
fi
```

### Fluxo recomendado de edição (Local + `rsync`)

Edite o código e os scripts `.sh` localmente (WSL/Linux + VS Code) e sincronize com `rsync -aP` ([NPAD Tutorial rsync](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/beginner/rsync.md)), evitando rodar extensões pesadas do VS Code Remote-SSH no nó de login do cluster. **Sempre exclua** `.venv/`, `.git/`, `build/`, `dist/`, `__pycache__/` e `*.so` ao sincronizar projetos Python para não ocupar a cota da Home nem sobrescrever extensões compiladas para o NPAD por binários incompatíveis do WSL (veja a função `npad-up` na [Seção 7](#7-atalhos-e-comandos-do-dia-a-dia)). Para edição gráfica remota leve no Linux, o NPAD também documenta montagem SFTP via `ssh://npad` no gerenciador de arquivos ([NPAD Gnome Files](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/beginner/gnome_files.md)).

---

## 2. Nó de Login (`service0`) vs. Nós de Computação

Ao conectar via SSH, você entra no **nó de login** (`service0.npad.internal`). Ele é compartilhado por todos os usuários e existe apenas para gerenciar arquivos, compilar códigos leves e submeter jobs ([NPAD Introdução Parte 2](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/beginner/introduction_part_2.md#L34-L60)).

- **Sem `sudo`:** Usuários não têm permissão de `root` ([NPAD Introdução Parte 1](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/beginner/introduction_part_1.md#L121-L123)).
- **Limite rígido de CPU no nó de login:** Processos no nó de login obedecem à regra $n_{\text{núcleos}} \times \text{uso} \times \text{tempo (min)} = 30$ ([Tabela de Uso do Nó de Login — NPAD](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/beginner/introduction_part_2.md#L36-L40)):

| Núcleos usados | Uso por núcleo | Tempo máximo antes de ser encerrado |
| :---: | :---: | :---: |
| 1 | 100% | 30 min |
| 5 | 100% | 6 min |
| 10 | 50% | 6 min |
| 10 | 100% | 3 min |

Todo processamento real deve ir para os **nós de computação** via `sbatch` ([Slurm `sbatch`](https://slurm.schedmd.com/sbatch.html)).

---

## 3. Partições do Cluster (`sinfo`) e Limites de Memória

Consulte o estado em tempo real das filas com `sinfo` ([NPAD Comandos SLURM](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/slurm_commands.md#L19-L103)):
- `idle`: Nó totalmente livre.
- `mix`: Nó parcialmente ocupado (aceita novos jobs nos núcleos/GPUs livres).
- `alloc`: Nó 100% ocupado.
- `down` / `drain` / `maint`: Indisponível ou em manutenção.

### Partições de CPU

| Partição | Limite de Tempo | RAM Padrão / CPU | RAM Máxima / CPU | Observações |
| :--- | :---: | :---: | :---: | :--- |
| `intel-128` | `20-00:00:0` | `2000 MB` | `4000 MB` | Ótima disponibilidade para testes e runs CPU MPI. |
| `intel-256` | `20-00:00:0` | `4000 MB` | `8000 MB` | Nós `r1i3n*` com 256 GB de RAM. |
| `intel-512` | `20-00:00:0` | `8000 MB` | `16000 MB` | Nós `r1i3n*` com 512 GB de RAM. |
| `amd-512` | `20-00:00:0` | `2000 MB` | `4000 MB` | Nós `r2n*`. |
| `amd-3tb` | `20-00:00:0` | — | — | Nós *fat* (`fat[01-07]`) para altíssima memória. |

*(Fonte dos limites padrão/máximo por CPU: [NPAD Introdução Parte 3](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/introduction_part_3.md#L81-L105); lista de partições observada via `sinfo` no cluster.)*

### Partições de GPU

Para alocar GPUs, utilize `#SBATCH --gpus-per-node=<N>` ([NPAD Uso da GPU](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/gpu.md#L1-L29) e [Slurm `--gpus-per-node`](https://slurm.schedmd.com/sbatch.html#OPT_gpus-per-node)):

| Partição | Nós (`sinfo`) | Limite de Tempo | GPUs por Nó | RAM / CPU (doc NPAD) | Observações práticas |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `gpu-8-v100` | `gpunode[0-1]` | `5-00:00:00` | 8x Tesla V100-SXM2-16GB | `8000 MB` | Driver NVIDIA 560.35.03 / CUDA 12.6 verificados no nó. |
| `gpu-4-a100` | `gpunode2` | `5-00:00:00` | 4x NVIDIA A100 | `2000 MB` | Como o limite por CPU é 2 GB, peça mais CPUs (`--cpus-per-task=6`+) para ter mais RAM de host ([NPAD GPU](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/gpu.md#L20-L25)). |
| `gpu-8-h100` | `gpunode3` | `5-00:00:00` | 8x NVIDIA H100 | — | Verificar disponibilidade (`STATE`) no `sinfo` antes de submeter. |
| `gpu-4-h200` | `gpunode[4-5]` | `5-00:00:00` | 4x NVIDIA H200 | — | GPUs de última geração do cluster. |

- **Isolamento automático de GPU:** O SLURM isola via `CUDA_VISIBLE_DEVICES` exatamente a quantidade requisitada em `--gpus-per-node` ([Slurm GRES Docs](https://slurm.schedmd.com/gres.html)). Ao pedir `--gpus-per-node=1`, `torch.cuda.device_count()` retorna `1`.

### Preempção e Backfill

- **Imunidade a preempção:** Jobs de usuários comuns têm imunidade garantida de **6 horas** em partições de GPU e **2 dias** nas demais partições. Acima desse limite, podem ser preemptados por jobs prioritários ([NPAD Introdução Parte 3](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/introduction_part_3.md#L54-L64)) — exatamente o cenário em que *checkpointing* tolerante a falhas com DeLIA é essencial.
- **Backfill:** Especificar um `--time` realista e curto permite que o escalonador encaixe o job em janelas livres à frente de jobs grandes na fila ([NPAD Introdução Parte 3](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/introduction_part_3.md#L205-L215)).

---

## 4. Arquitetura de Armazenamento (`~` vs. `~/scratch`)

| Critério | **Home (`~`)** | **Scratch Global (`~/scratch` / `$SCRATCH_GLOBAL`)** |
| :--- | :--- | :--- |
| **Caminho real** | `/home/<SEU_USUARIO>` | `/scratch/global/<SEU_USUARIO>` (montado no BeeGFS `/mnt/beegfs`) |
| **Persistência** | Permanente (nunca é apagada por inatividade). | Temporária: arquivos inativos por longos períodos podem ser removidos pela equipe do NPAD ([NPAD Introdução Parte 3](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/introduction_part_3.md#L202)). |
| **Cota** | **100 GB (soft) / 1 TB (hard)**, arquivo `~/.npadquota`. | Pool compartilhado de **112 TB** com cota individual **`unlimited`** (`beegfs-ctl`). |
| **Tecnologia** | Rede compartilhada (NFS). | Sistema de arquivos paralelo [**BeeGFS**](https://doc.beegfs.io/latest/overview/overview.html) otimizado para E/S concorrente via MPI. |
| **Uso neste projeto** | Repositório Git, scripts `.py`/`.sh`, relatórios `.json`/`.png`. | Checkpoints do DeLIA (`ft_dir`), datasets grandes, caches (`POETRY_CACHE_DIR`). |

### Como ler o `~/.npadquota` e o espaço do Scratch

O arquivo `~/.npadquota` segue o formato padrão de cotas de disco Linux (3 colunas de blocos em KiB + 3 colunas de inodes/arquivos, conforme [`quota(1)`](https://man7.org/linux/man-pages/man1/quota.1.html)):

```text
<USADO_KB>  <SOFT_KB=104857600 (100 GB)>  <HARD_KB=1073741824 (1 TB)>  <ARQUIVOS>  <SOFT_ARQ=0>  <HARD_ARQ=0>
```

Para consultar o uso do Scratch paralelo ([`df(1)`](https://man7.org/linux/man-pages/man1/df.1.html) e [BeeGFS Quota Docs](https://doc.beegfs.io/latest/advanced_topics/quota.html)):

```bash
df -h ~/scratch/
beegfs-ctl --getquota --uid $USER
```

---

## 5. Módulos (`module`), Conda, Poetry e Build do `deliapy`

### Módulos úteis no cluster (`module avail`)

Principais módulos para este projeto ([NPAD Introdução Parte 3](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/introduction_part_3.md#L118-L153)):
- **Compilador C/C++:** `module load compilers/gnu/14.2.0` (padrão `D`) ou `compilers/gnu/11.2`.
- **OpenMPI 5.0.10 (UCX):** `module load libraries/openmpi/5.0.10-gnu14-ucxmt` (padrão `D`, em `/opt/npad/shared/libraries/openmpi/5.0.10-gnu14-ucxmt`).
- **CUDA:** `compilers/nvidia/cuda/12.6` (padrão `D`) e `libraries/cudnn-cuda12/9.6.0`.
- **Python via módulo:** apenas `softwares/python/3.10.5-gnu8` e `softwares/python/3.13.7-gnu8` (não atendem ao requisito `>=3.11,<3.12` do `deliapy`).

### Ambientes Conda e Poetry

O NPAD disponibiliza ambientes compartilhados em `/opt/npad/shared/conda/` (`geral` com Python 3.7.11, `gpu` com PyTorch 2.0.1 + CUDA, `forge`, `forge2023`, etc., listados via `conda env list` — [NPAD Tutorial Conda](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/conda.md#L11-L39)).

Para rodar Python 3.11 com Poetry e extensões compiladas pelo GCC 14 neste projeto:

1. **Crie um ambiente Conda próprio com `--override-channels` e `libstdcxx-ng>=14`:**
   Sem `--override-channels`, o resolvedor do Conda pode misturar canais padrão e travar no nó de login; incluir `libstdcxx-ng>=14` evita erros de `GLIBCXX_3.4.32 not found` ao importar extensões C++17 compiladas com `compilers/gnu/14.2.0` ([NPAD Tutorial Conda](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/conda.md#L85-L108)):
   ```bash
   conda create -n py311 -c conda-forge --override-channels python=3.11 pip "libstdcxx-ng>=14" "libgcc-ng>=14" -y
   ```
2. **Direcione caches para o Scratch e desative virtualenvs extras do Poetry** ([Poetry Configuration Docs](https://python-poetry.org/docs/configuration/#cache-dir)):
   Se o Poetry criar um `.venv` separado, chamar `conda activate py311` + `mpirun python ...` no SLURM resultará em `ModuleNotFoundError`. Desative a criação de venvs separados para o `poetry install` instalar direto dentro do `py311`:
   ```bash
   export PIP_CACHE_DIR="${SCRATCH_GLOBAL:-$HOME/scratch}/.cache/pip"
   export POETRY_CACHE_DIR="${SCRATCH_GLOBAL:-$HOME/scratch}/.cache/pypoetry"
   eval "$(conda shell.bash hook)"
   conda activate py311
   pip install poetry
   poetry config virtualenvs.create false
   ```
3. **Ordem de ativação nos scripts `#SBATCH`:**
   Sempre ative o Conda **antes** de carregar os módulos do NPAD ([NPAD Tutorial Conda](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/conda.md#L73-L83)) para manter o `mpirun` e as bibliotecas GCC/OpenMPI do cluster no início de `PATH` e `LD_LIBRARY_PATH`:
   ```bash
   eval "$(conda shell.bash hook)"
   conda activate py311
   module load compilers/gnu/14.2.0 libraries/openmpi/5.0.10-gnu14-ucxmt
   ```

### Compilando `mpi4py` e `deliapy` para o NPAD

Wheels pré-compilados de `deliapy` no PyPI/TestPyPI falham no NPAD porque:
1. São compilados em imagens Debian com `glibc` mais nova que a **`glibc 2.28`** do RHEL/CentOS 8 (`el8`) no NPAD ([NPAD `install_apps.md`](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/install_apps.md#L112)), além de embutirem **RPATH** absoluto para `/opt/openmpi-5.0.8/lib` com `-Wl,--disable-new-dtags` ([`ld.so(8)`](https://man7.org/linux/man-pages/man8/ld.so.8.html)).
2. O pacote fonte (`sdist`) do `deliapy` exclui `deliapy/**/*.pyx`, `deliapy/**/*.pxd` ([`pyproject.toml`](https://gitlab.com/lappsufrn/deliapy/-/blob/main/pyproject.toml)) e o submódulo `deliacpp/`, portanto `pip install --no-binary :all: deliapy` via PyPI/TestPyPI não consegue compilar.

Execute a compilação abaixo (de preferência dentro de um job rápido na fila `intel-128` para não sobrecarregar o nó de login `service0`):

```bash
eval "$(conda shell.bash hook)"
conda activate py311
module load compilers/gnu/14.2.0 libraries/openmpi/5.0.10-gnu14-ucxmt

export MPI_HOME=/opt/npad/shared/libraries/openmpi/5.0.10-gnu14-ucxmt
export PIP_CACHE_DIR="${SCRATCH_GLOBAL:-$HOME/scratch}/.cache/pip"

# 1. Instalar ferramentas de build e NumPy antes (necessário para --no-build-isolation)
pip install "setuptools>=80.0.0" wheel "Cython>=3.0,<4.0" "numpy<2.0"

# 2. Compilar mpi4py contra o OpenMPI 5.0.10 do NPAD
pip install --no-build-isolation "mpi4py>=3.1.6"

# 3. Clonar o deliapy com o submódulo C++ (deliacpp) e compilar direto no py311
git clone --recurse-submodules https://gitlab.com/lappsufrn/deliapy.git "${SCRATCH_GLOBAL:-$HOME/scratch}/src_deliapy"
cd "${SCRATCH_GLOBAL:-$HOME/scratch}/src_deliapy"
pip install --no-build-isolation --no-deps .
```

*(Nota: Caso utilize `poetry install` no repositório do experimento, instale ou reinstale o `deliapy` compilado acima **após** o `poetry install` — ou aponte a dependência para `${SCRATCH_GLOBAL:-$HOME/scratch}/src_deliapy` — para evitar que o Poetry substitua a extensão compilada por um wheel incompatível do PyPI/TestPyPI.)*

*(Observação para código Python deste repositório: antes de importar `mpi4py` ou `deliapy`, pré-carregue `libmpi.so` via `ctypes.CDLL(os.path.join(os.environ.get("MPI_HOME", "/opt/npad/shared/libraries/openmpi/5.0.10-gnu14-ucxmt"), "lib/libmpi.so"), mode=ctypes.RTLD_GLOBAL)` e, ao reduzir gradientes via MPI na GPU, converta tensores com `.detach().cpu().numpy()` em vez de `.numpy()` direto na VRAM.)*

---

## 6. Anatomia de Scripts `.sh` para o SLURM e Templates Prontos

Todo script de submissão segue **4 blocos** na ordem exata:
1. **Linha 1 — Shebang (`#!/bin/bash`):** obrigatório na primeira linha ([`execve(2)`](https://man7.org/linux/man-pages/man2/execve.2.html)).
2. **Bloco `#SBATCH`:** diretivas de alocação ([Slurm `sbatch`](https://slurm.schedmd.com/sbatch.html)). Podem haver linhas em branco ou comentários entre elas, mas **nenhum comando executável antes**. Se usar subpasta em `--output`/`--error` (ex: `logs/%x_%j.out`), crie a pasta **antes** do `sbatch` (`mkdir -p logs`), caso contrário o job falha antes de iniciar ([Slurm FAQ](https://slurm.schedmd.com/faq.html)).
3. **Preparação (roda 1 vez):** ativação do Conda **antes** de `module load`, criação de diretórios isolados por job no `$SCRATCH_GLOBAL` e exportação de `OMP_NUM_THREADS`.
4. **Execução:** para programas **OpenMPI 5.0.10** (`mpi4py`/`deliapy`), utilize **`mpirun`** do módulo carregado (sem `-np`, pois o OpenMPI lê `$SLURM_NTASKS` e `$SLURM_JOB_NODELIST` automaticamente — [NPAD Tutorial MPI](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/advanced/mpi.md#L106-L109)), evitando conflitos entre o `pmix/3.1.4` do sistema e o PMIx v5 do OpenMPI 5.

### Template 1: Job MPI Multi-Nó em CPU

```bash
#!/bin/bash

#SBATCH --job-name=delia_cpu_mpi
#SBATCH --partition=intel-128
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000
#SBATCH --time=0-00:30
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Caso prefira salvar em subpasta (ex: logs/%x_%j.out), execute 'mkdir -p logs' ANTES do sbatch.

set -e
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

eval "$(conda shell.bash hook)"
conda activate py311
module load compilers/gnu/14.2.0 libraries/openmpi/5.0.10-gnu14-ucxmt

CKPT_DIR="${SCRATCH_GLOBAL:-$HOME/scratch}/deliapy-hpc-llm-train/checkpoints/job_$SLURM_JOB_ID"
mkdir -p "$CKPT_DIR"

# Exemplo de execução MPI (substitua pelo script/módulo do experimento;
# se houver pós-processamento após o mpirun, considere o Signal 11 benigno de teardown do deliapy)
mpirun python -m src --mode delia --ft-dir "$CKPT_DIR"
```

### Template 2: Job Multi-GPU (PyTorch + MPI)

```bash
#!/bin/bash

#SBATCH --job-name=delia_gpu_train
#SBATCH --partition=gpu-8-v100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Caso prefira salvar em subpasta (ex: logs/%x_%j.out), execute 'mkdir -p logs' ANTES do sbatch.

set -e
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

eval "$(conda shell.bash hook)"
conda activate py311
module load compilers/gnu/14.2.0 libraries/openmpi/5.0.10-gnu14-ucxmt

CKPT_DIR="${SCRATCH_GLOBAL:-$HOME/scratch}/deliapy-hpc-llm-train/checkpoints/job_$SLURM_JOB_ID"
mkdir -p "$CKPT_DIR"

nvidia-smi
# Exemplo de execução MPI em GPU (substitua pelo script/módulo do experimento)
mpirun python -m src --mode delia --device cuda --ft-dir "$CKPT_DIR"
```

---

## 7. Atalhos e Comandos do Dia a Dia

### Comandos nativos do SLURM ([NPAD Comandos SLURM](https://github.com/NPAD-UFRN/Tutorials/blob/main/docs/intermediate/slurm_commands.md))

| Comando | Função |
| :--- | :--- |
| `sinfo` | Lista partições, limites de tempo e estado dos nós (`idle`, `mix`, `alloc`, `down`). |
| `sbatch script.sh` | Envia o script para a fila de computação. |
| `squeue -u $USER` | Lista apenas os seus jobs na fila (`PD` = aguardando, `R` = executando). |
| `scancel <JOBID>` | Cancela um job pendente ou em execução. |
| `sacct -j <JOBID>` | Consulta o histórico, nós usados, CPUs alocadas e `ExitCode` após o término ([Slurm `sacct`](https://slurm.schedmd.com/sacct.html)). |

### Atalhos locais recomendados (máquina local — `~/.zshrc`)

```zsh
# Envia arquivo/pasta local para o NPAD excluindo ambientes e binários locais (destino padrão: ~/testes_slurm/)
npad-up() {
    local dest="${2:-testes_slurm/}"
    rsync -aP \
        --exclude='.venv/' \
        --exclude='.git/' \
        --exclude='__pycache__/' \
        --exclude='build/' \
        --exclude='dist/' \
        --exclude='*.so' \
        "$1" "npad:~/$dest"
}

# Baixa arquivo/pasta da Home do NPAD para o PC local (destino padrão: .)
npad-down() {
    rsync -aP "npad:~/$1" "${2:-.}"
}
```

### Atalhos remotos configurados (NPAD — `~/.bashrc`)

```bash
alias c='clear'
alias squ='squeue -u "$USER"'

# Submete o script e já mostra a fila do usuário
sb() {
    sbatch "$@" && squeue -u "$USER"
}

# Exibe a cota da Home convertida em GB (ou bruta com cota-raw)
cota() {
    awk '{
        printf "%-14s %-14s %-14s %-10s\n", "USADO", "COTA_SOFT", "COTA_HARD", "ARQUIVOS";
        printf "%-14s %-14s %-14s %-10s\n", $1" KB", ($2/1048576)" GB", ($3/1048576)" GB", $4
    }' ~/.npadquota
}

alias cota-raw='(echo "USADO(KB) SOFT(KB) HARD(KB) ARQUIVOS SOFT_ARQ HARD_ARQ"; cat ~/.npadquota) | column -t'
```
