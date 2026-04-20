# scheduling_fucho_games

> POC exploring CP-SAT (OR-Tools) and Genetic Algorithm (DEAP) solutions to an
> Integer Linear Programme for round-robin football match scheduling.

---

## Project Layout

```
scheduling_fucho_games/          ← git repo root
├── .venv/                       ← virtualenv (uv-managed)
├── pyproject.toml
├── uv.lock
├── README.md
└── scheduling_fucho_games/      ← Python package
    ├── __init__.py
    ├── __main__.py              ← python -m entrypoint
    ├── main.py                  ← argparse CLI + output serialisation
    ├── model.py                 ← TournamentProblem, Team, Match, VenueSlot, SolverResult
    ├── yml_loader.py            ← YAML → TournamentProblem (validated)
    ├── utils.py                 ← pure: date expansion, holidays, match generation
    ├── algorithms/
    │   ├── cpsat.py             ← OR-Tools CP-SAT solver
    │   └── genetic_algorithm.py ← DEAP GA solver
    └── assets/
        ├── example_input.yml   ← canonical input schema
        └── example_output.yml  ← solver output (generated)
```

---

## Problem Definition

### Sets

| Symbol | Definition |
|--------|-----------|
| $\mathcal{T}$ | Teams, $\lvert\mathcal{T}\rvert = K$ (must be even) |
| $\mathcal{M}$ | All unordered pairs $(i,j),\; i < j$ — total $\binom{K}{2}$ matches |
| $\mathcal{D}$ | Matchdays $0, \ldots, K{-}2$ |
| $\mathcal{V}$ | Venue-slots $= \mathcal{F} \times \mathcal{S}_f$ (over all fields $f$) |
| $r_v$ | Revenue of venue-slot $v \in \mathcal{V}$ |
| $\mathcal{A}$ | Feasible calendar dates after availability and exclusion filtering |

### Decision Variable

$$x_{m,d,v} \in \{0,1\}, \quad \forall\; m \in \mathcal{M},\; d \in \mathcal{D},\; v \in \mathcal{V}$$

$x_{m,d,v} = 1$ iff match $m$ is played on matchday $d$ using venue-slot $v$.

### Objective

$$\max \sum_{m \in \mathcal{M}} \sum_{d \in \mathcal{D}} \sum_{v \in \mathcal{V}} r_v \cdot x_{m,d,v}$$

### Constraints

**C1 — Each match scheduled exactly once:**
$$\sum_{d,v} x_{m,d,v} = 1 \quad \forall\; m$$

**C2 — Each team plays exactly once per matchday:**
$$\sum_{m \ni t,\; v} x_{m,d,v} = 1 \quad \forall\; t \in \mathcal{T},\; d \in \mathcal{D}$$

**C3 — No venue-slot double-booked per matchday:**
$$\sum_m x_{m,d,v} \leq 1 \quad \forall\; d, v$$

**C4 — Exactly $K/2$ matches per matchday:**
$$\sum_{m,v} x_{m,d,v} = K/2 \quad \forall\; d$$

**C5 — Matchday gap $\geq \Delta$ days** (enforced via pre-selected calendar dates).

---

## Scope

- **Format**: single round-robin ($\binom{K}{2}$ total matches, $K{-}1$ matchdays).
- **Solvers**: CP-SAT (exact, globally optimal) and Genetic Algorithm (metaheuristic).
- **Output**: best + second-best solution exported to YAML.
- **Infeasibility**: explicit diagnostic messages with corrective guidance.

---

## Setup (uv)

```bash
# From repo root
uv venv                     # create .venv
uv pip install -e .         # editable install with all deps
```

---

## Usage

All commands run from the **repo root**.

```bash
# CP-SAT solver (exact, optimal)
.venv/bin/python -m scheduling_fucho_games \
    --config scheduling_fucho_games/assets/example_input.yml \
    --algorithm cpsat \
    --output scheduling_fucho_games/assets/example_output.yml \
    --timeout 60

# Genetic Algorithm solver (metaheuristic)
.venv/bin/python -m scheduling_fucho_games \
    --config scheduling_fucho_games/assets/example_input.yml \
    --algorithm genetic \
    --output scheduling_fucho_games/assets/example_output.yml \
    --generations 600 \
    --pop-size 300 \
    --seed 42

# All CLI flags
.venv/bin/python -m scheduling_fucho_games --help
```

---

## Input Schema (`scheduling_fucho_games/assets/example_input.yml`)

```yaml
format: "round_robin"

teams:
  - { id: 1, name: "Team 1" }
  # …

fields:
  - id: 1
    name: "Field 1"
    time_slots:
      - { range: "20:00-21:00", revenue: 100 }

constraints:
  min_matchday_gap_days: 7
  inclusion:
    start_date: "2022-01-01"           # YYYY-MM-DD
    end_date:   "2022-03-01"
    days_available: ["Friday", "Saturday", "Sunday"]
  exclusion:
    holidays: false                    # true → exclude official holidays
    holiday_country: "MX"             # ISO 3166-1 alpha-2
    specific_dates: []
```

---

## Output Schema

```yaml
algorithm: cpsat
status: OPTIMAL                        # OPTIMAL | FEASIBLE | INFEASIBLE | NOT_CONVERGED
solve_time_s: 0.05
infeasibility_reason: null
best_solution:
  total_revenue: 1600
  schedule:
    - matchday: 1
      date: "2022-01-01"
      matches:
        - { match: "Team 1 vs Team 2", field: "Field 1", time_slot: "22:00-23:00", revenue: 110 }
        # …
second_best_solution:
  total_revenue: 1590
  schedule: …
```

---

## Feasibility Requirements

For $K$ teams:
1. $\geq K{-}1$ calendar dates available with mutual gap $\geq \Delta$.
2. $\geq K/2$ venue-slots total (fields × time-slots).
3. $K$ must be even (no bye-team support in v0.1).

---

## Algorithm Details

### CP-SAT (OR-Tools)
- Full Boolean IP with `CpModel`. `CpSolverSolutionCallback` captures improving incumbents → top-2 = best + second-best.

### Genetic Algorithm (DEAP)
- **Chromosome**: list of $M$ integers; `gene[m] = d × V + v`.
- **Fitness**: `revenue − 10⁵ × (venue_conflicts + team_conflicts)`.
- **Operators**: two-point crossover, uniform-int mutation, tournament selection ($k=3$).
- **Hall of Fame** size 2 → best + second-best feasible individuals.
