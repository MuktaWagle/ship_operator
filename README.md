# ShipNaviSim — Manual Fleet Operator Demo

A shared-world vessel simulation for manually steering, stopping and releasing ships while monitoring their separation. Autonomous motion uses a route-following controller; this demo does **not** run a trained RL or behavioural-cloning policy.

## Setup and run (Windows / Conda)

Open a terminal in this folder. For your existing environment:

```bat
conda activate maritime-rl
python -m pip install -r requirements.txt
python demo_manual_operator.py
```

For a new environment:

```bat
conda env create -f environment.yml
conda activate maritime-rl
python demo_manual_operator.py
```

Python 3.10 is the reference version. The dependency files contain only the libraries required for this demo. The optional legacy `create_minari_dataset()` helper in `ship_env.py` requires a separate Minari installation; dataset export is outside this demo's scope.

## Controls

The window starts paused. Select a ship, then press SPACE.

| Key / action | Behaviour |
|---|---|
| TAB | Select the next active ship |
| Click a panel row | Select that ship |
| LEFT / A | Move while turning left for the chosen duration |
| RIGHT / D | Move while turning right for the chosen duration |
| S | Hold position and heading for the chosen duration |
| R | Release the selected operator-controlled ship early |
| G | Toggle Greedy Auto-Intervention Dispatcher |
| UP / DOWN | Change the duration for the next command in 10-second increments |
| SPACE | Pause/resume simulation time and command timers |
| ESC / close window | End the demonstration |

Only one ship can be under manual control at a time. Another command for the same ship replaces its command and restarts the timer. Release or timer expiry frees the operator and starts automatic route recovery. Recovering vessels do not occupy the operator, so another ship can be controlled while recovery continues.

Stop is an immediate hold in this simplified model. Movement resumes through recovery; a ship already on its route can rejoin immediately. Turning continues throughout a left/right command: there is no selectable total-angle command yet. Commands affect movement on simulation steps, with no separate operator reaction delay.

## Current settings

Edit the constants near the top of `demo_manual_operator.py`.

| Setting | Current value | Meaning |
|---|---:|---|
| `FLEET_SIZE` | 10 | Number of selected source routes |
| `SCENARIO_SEED` | 43 | Reproducible route selection |
| `DT` | 10 | Simulated seconds per step; also passed to preprocessing |
| `STEPS_PER_REAL_SECOND` | 3 | Target display stepping rate |
| `INITIAL_COMMAND_DURATION` | 100 | Initial command duration in simulated seconds |
| `MAX_COMMAND_DURATION` | 240 | Maximum keyboard-selectable duration |
| `TURN_RATE_DEG` | 1.0 | Degrees turned per simulated second |
| `WARNING_DISTANCE_M` | 500 | Live warning threshold in metres |
| `MIN_START_DISTANCE_M` | 500 | Required separation of selected starting positions |
| `MAX_STEPS` | 1000 | Episode step limit |

At the current target rate, 30 simulated seconds pass per real second: a 100-second command is visible for about 3.33 real seconds. Set `STEPS_PER_REAL_SECOND = 0.5` for slower viewing: the same command takes about 20 real seconds. Rendering load can make playback slower. Changing playback rate does not change manoeuvre duration in simulation time.

Keep DT at 10 for the existing demonstration. Changing command duration changes the manoeuvre itself: at 1 degree/second, a 100-second turn commands 100 degrees of rotation. UP/DOWN changes only the next command's duration.

## Data and fleet layouts

Input: `raw_data/synthetic_ais_data.csv` (included). The loader filters the configured longitude/latitude region, valid headings and status 0; sorts records; splits journeys at gaps over 30 minutes; and interpolates routes at DT-second intervals. The scenario builder selects sufficiently long routes with the requested starting separation, preserves their coordinates and aligns their start times. Simulated IDs start at 10000; the terminal reports each selected source trajectory index.

This is synthetic simultaneous traffic, not evidence that these ships travelled together at those times. Source CSV provenance beyond the supplied project is not independently verified.

Change the seed for another reproducible selection. Increase starting separation to reduce initial clustering, or reduce fleet size if selection fails. The selection search tries up to 200 random orderings; failure does not prove no valid selection exists. Starting separation does not guarantee uniform coverage or prevent later convergence. There is no generator guaranteeing only two or three close encounters.

## Distance display and warnings

All pairs of active ships are checked, including stopped ships. Ten active ships have 45 pairs.

| Measurement | Display / use |
|---|---|
| Current centre separation | Closest active pair on the map; nearest active neighbour in each row |
| Minimum during the last step | Orange notice for an encounter below the threshold |
| Minimum since episode start | Per-pair records; the ten closest records print on exit |

Pairs **strictly below** the threshold produce red ship rings that blink every half-second of real time, red row text and warning counts. The terminal reports entry/exit from the warning zone and close encounters between step endpoints. Warnings update with simulation steps; blinking continues while paused. The last-step notice remains until the next step.

Last-step minima assume linear motion between step endpoints. They are retrospective measurements, not predictions. The 500 m threshold is an experimental centre-distance setting, not a collision definition or universal safety distance. Alerts do not steer, stop or select vessels automatically.

## Files

| File | Purpose |
|---|---|
| `demo_manual_operator.py` | Main Pygame interface, manual commands and live alerts |
| `scenario.py` | Shared trajectory loading and fleet selection |
| `route_controller.py` | Route following, timed turns/hold and route recovery |
| `fleet_simulator.py` | Shared clock/state, movement, retirement and pair distances |
| `ship_env.py` | Underlying vessel environment and coordinate conversion |
| `utils.py`, `split_chunks.py` | Supporting calculations and data preprocessing |
| `raw_data/synthetic_ais_data.csv` | Source data |
| `star-icon.jpg` | Asset retained for the underlying environment renderer |
| `operator_tracker.py` | Operator status (Available/Preparing/Assisting), budget tracker, action masking |
| `allocation_env.py` | Gymnasium state-action-reward environment for learning operator allocation |
| `test_allocation_env.py` | Checks for allocation observations, action masks and event rewards |
| `demo_operator_budget.py` | Pygame interface with live operator HUD and budget tracker |
| `run_simulation_test.py` | Headless terminal simulation test of fleet and operator budget |
| `test_operator_tracker.py` | Unit and integration tests for operator & budget tracker |
| `test_fleet_reset.py`, `test_fleet_endings.py` | Existing simulator checks |
| `requirements.txt`, `environment.yml` | Installation dependencies |
| `LICENSE` | Original project licence, retained unchanged |

## Simulator interface for teammates

Construct `FleetSimulator` using trajectories (columns x, y, speed, heading), timestamps, unique vessel IDs, region, dt and maximum steps. Timestamps must start together and match dt.

```python
state = fleet.reset(seed=43)
controller = RouteController(fleet)
while not fleet.finished:
    actions = controller.actions(state)
    state = fleet.step(actions)
fleet.close()
```

`step()` accepts exactly one `[dx, dy, dheading]` action per active ship: displacement in metres and heading change in radians. Speed is displacement magnitude divided by dt. Actions must satisfy the environment bounds. `[0, 0, 0]` holds position and heading.

Use `get_state()`, `active_ship_ids`, `finished`, `get_distance_metrics()` and `close()`. State contains `timestamp`, `step` and a `ships` mapping. Ship entries include position, speed, heading, timestamp, goal and status. Statuses are `active`, `arrived`, `out of bounds`, and `time limit`.

Arrival means distance to goal is less than 200 m. Retired vessels keep their final state but leave active traffic and stop receiving commands. Ships that finish within a step are included in that step's distance measurements. Reset clears distance metrics. Closing the demo early can leave ships marked active; that alone is not an error.

`RouteController.take_over(ship_id, direction, duration, turn_rate_deg)` accepts `left`, `right`, or `stop`. `release(ship_id)` initiates recovery. Duration must be positive, finite and a multiple of dt. The one-operator restriction is enforced by the demo, not by `RouteController` itself; integrations must enforce it too.

Recovery projects onto the remaining reference route and navigates from the ship's actual location; it does not teleport the ship. Cruise speed is the route's median source speed clipped to 3–8 m/s. Recovery can rotate in place. These are prototype kinematics, not realistic ship dynamics.

## Operator state and budget tracker interface

`OperatorBudgetTracker` tracks operator availability, communication delays ($\delta$), intervention duration ($\tau$), and assistance budget ($B$).

```python
from operator_tracker import OperatorBudgetTracker, OperatorStatus

tracker = OperatorBudgetTracker(
    vessel_ids=fleet.vessel_ids,
    total_budget=300,   # Total budget B in seconds
    duration=60,        # Intervention duration tau in seconds
    delay=10,           # Communication buffer delay delta in seconds
    dt=fleet.dt,        # Simulator step in seconds
)

# 1. Action Masking (for Task 4 DQN / Task 6 baselines)
# Returns boolean array of shape (N + 1,) for actions 0..N
mask = tracker.get_action_mask(fleet.active_ship_ids)

# 2. Dispatching an intervention decision a_t in {0, ..., N}
# a_t = 0 is no-op; a_t = i assigns vessel i
tracker.dispatch(action=i, active_ship_ids=fleet.active_ship_ids)

# 3. Reading step metrics for Task 3 reward calculation (r_t = ... - w_h * I_t)
i_t = tracker.interventions_initiated_this_step  # 1 if intervention initiated, else 0

# 4. Advancing the tracker with the fleet step
res = tracker.step(active_ship_ids=fleet.active_ship_ids)

# 5. Obtaining the operator observation vector for RL policies
obs_vec = tracker.get_observation_vector(normalize=True)  # shape (7,)
```

State variables tracked:
- `b_t`: Remaining assistance budget in seconds (`tracker.remaining_budget`).
- `h_t`: Operator status (`OperatorStatus.AVAILABLE`, `PREPARING`, `ASSISTING`).
- `T_t`: Elapsed simulation time in seconds (`tracker.simulation_time`).
- `I_t`: Interventions initiated indicator on current step (`tracker.interventions_initiated_this_step`).

Transition lifecycle:
1. $a_t = i$ dispatched $\to$ operator status becomes `PREPARING` with buffer timer $\delta$.
2. When buffer $\delta$ expires $\to$ status becomes `ASSISTING` for duration $\tau$.
3. While `ASSISTING` $\to$ budget $b_t$ is consumed by $dt$ each step.
4. When $\tau$ expires or budget reaches 0 $\to$ status returns to `AVAILABLE`.
5. If the assisted vessel arrives/retires early $\to$ operator is immediately freed.

## Checks & Testing

### Allocation state-action-reward environment

`FleetAllocationEnv` wraps `FleetSimulator`, `RouteController` and
`OperatorBudgetTracker` as a Gymnasium environment. It uses one proactive
allocation decision per fleet simulation step (`dt`, currently 10 simulated
seconds):

```python
from allocation_env import FleetAllocationEnv

env = FleetAllocationEnv(fleet, controller, tracker)
observation, info = env.reset(seed=43)

while True:
    # 0 = idle; 1..N = assign the corresponding vessel.
    valid_actions = info["action_mask"]
    action = 0
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

The observation keeps a fixed shape for the whole episode. Retired vessels
remain in the vessel tensor with `active=0`. Ship features contain normalized
position, speed, heading (`sin`/`cos`), timestamp, active flag and assigned
flag. The operator vector contains budget, one-hot operator status,
preparation time, remaining assistance time, elapsed assistance time and
simulation time. `info["action_mask"]` identifies valid discrete actions.

Rewards use event deltas for the current step:

```text
reward = wg * newly_completed_journeys
       - wc * newly_started_collisions
       - wm * newly_started_near_misses
       - wh * accepted_interventions
```

The default collision threshold is 50 m and the default near-miss threshold
is 500 m. These are configuration values, not physical safety definitions.
The allocation environment currently uses a configurable fixed assistance
maneuver (`right` by default); replacing that maneuver with a learned
assistance controller is a separate phase.

Run its checks with:

```bash
python test_allocation_env.py
```

1. **Unit & Integration Tests**:
```bash
python test_operator_tracker.py
```

2. **Headless Terminal Simulation Test**:
Runs a full 30-step simulation log showing step-by-step budget consumption, buffer delay, and action masking:
```bash
python run_simulation_test.py
```

3. **Pygame Visual Simulator Demo**:
Launches the fleet simulator with the live Operator State & Budget HUD (status badges, live budget progress bar, manual or greedy auto-dispatch):
```bash
python demo_operator_budget.py
```

4. **Existing Simulator Regression Checks**:
```bash
python test_fleet_reset.py
python test_fleet_endings.py
```

## Scope and cleanup

The maintained entry point is `demo_manual_operator.py`. Shared preprocessing was extracted from `demo_operator_recovery.py` into `scenario.py`, so the manual demo no longer depends on an older demo script. Runtime messages now use the actual fleet size and include the stop key.

Removed from this cleaned package: `archive/`, obsolete `demo_fleet_manager.py` and `demo_operator_recovery.py`, legacy training scripts `IL-BC.py` and `det_bc.py`, legacy dataset/replay script `test_env.py`, the stray `conda` file, Python caches, and the superseded `Theja_README_fleet.md`. The original long training environment/dependencies were replaced by demo-focused installation files. The original upload remains recoverable through file version history.

Core simulator/controller behaviour and your selected settings are retained. Minari is imported only when its optional export helper is called. Further work includes selectable turn angles (deferred), deliberate encounter layouts, timestamped intervention logs, trained policies, operator allocation, and a fleet training wrapper. None of these is claimed as implemented here.
