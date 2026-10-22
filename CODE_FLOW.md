# Ship Operator Low-Level Code Flow

This document describes the runtime path after the project restructuring. The
two demos are user-facing entry points; the reusable simulator and operator
allocation code lives under `src/`.

## Component flow

```mermaid
flowchart LR
    subgraph EntryPoints[Entry points]
        Manual[demo/demo_manual_operator.py]
        Budget[demo/demo_operator_budget.py]
        AllocationDemo[demo/demo_allocation.py]
        Test[test/run_bc_allocation_test.py]
    end

    subgraph Navigation[BC navigation]
        Scenario[bc_navigation/scenario.py\nload_bc_scenario]
        Data[(journeys.npz\nmanifest.json)]
        Policy[bc_navigation/controller.py\nFrozenBCPolicy + BCController]
        Model[bc_navigation/model.py\nBCPolicy neural network]
    end

    subgraph Runtime[src/]
        Allocation[src/allocation_env.py\nFleetAllocationEnv]
        Tracker[src/operator_tracker.py\nOperatorBudgetTracker]
        Fleet[src/fleet_simulator.py\nFleetSimulator]
        Ship[src/ship_env.py\nShipEnvironment]
        Utils[utils.py\ncoordinate and angle helpers]
    end

    Manual --> Scenario
    Budget --> Scenario
    AllocationDemo --> Scenario
    Test --> Scenario
    Scenario --> Data
    Scenario --> Fleet
    Manual --> Policy
    Budget --> Policy
    AllocationDemo --> Policy
    Test --> Policy
    Policy --> Model
    Policy --> Fleet

    Manual --> Tracker
    Budget --> Tracker
    AllocationDemo --> Allocation
    Test --> Allocation
    Allocation --> Tracker
    Allocation --> Policy
    Allocation --> Fleet
    Fleet --> Ship
    Ship --> Utils
```

The grey reference line in the demos is drawn from the prepared trajectories
returned by `load_bc_scenario`. It is visualization-only. `BCController`
chooses actions from the live ship state and goal; it does not consume the
reference waypoints.

## One proactive allocation step

```mermaid
sequenceDiagram
    participant P as Allocation policy
    participant E as FleetAllocationEnv
    participant O as OperatorBudgetTracker
    participant C as BCController
    participant F as FleetSimulator
    participant S as ShipEnvironment instances

    P->>E: action a_t in {0..N}
    E->>O: dispatch(action, active_ship_ids)
    O-->>E: status, budget, action validity

    alt operator is ASSISTING
        E->>C: take_over(ship_id, maneuver, duration)
    else operator is AVAILABLE or PREPARING
        Note over C: autonomous BC mode remains active
    end

    E->>C: actions(current_state)
    alt ship is operator-controlled
        C-->>E: timed manual action [dx, dy, d_heading]
    else ship is autonomous
        C->>C: policy.predict(ship state, goal)
        C-->>E: BC action [dx, dy, d_heading]
    end

    E->>F: step(actions for every active ship)
    F->>S: validate and apply each action
    S-->>F: next ship states and statuses
    F-->>E: next fleet state, distances, events
    E->>O: step(active_ship_ids)
    O-->>E: updated budget and operator state
    E->>E: calculate event deltas and reward
    E-->>P: observation, reward, done, info/action_mask
```

## Responsibilities and boundaries

| Component | Responsibility | Does not do |
|---|---|---|
| `load_bc_scenario` | Select valid, synchronized BC journeys and build simulator inputs | Choose navigation actions during the episode |
| `FrozenBCPolicy` | Load the frozen checkpoint and produce turn/speed predictions | Manage ships, budgets, or interventions |
| `BCController` | Convert BC predictions or operator takeover commands into simulator actions | Decide which ship should receive assistance |
| `FleetAllocationEnv` | Expose fixed-shape observations, discrete allocation actions, event rewards, and masks | Train an RL algorithm itself |
| `OperatorBudgetTracker` | Enforce one-operator availability, delay, duration, budget, and intervention metrics | Move ships or calculate navigation commands |
| `FleetSimulator` | Maintain one shared clock, validate actions, update all ships, and measure fleet events | Decide allocation actions |
| `ShipEnvironment` | Store per-ship state, goals, bounds, observations, and retirement status | Coordinate the operator budget |

## State, action, and reward path

`FleetAllocationEnv` produces three observation groups:

- `ships`: one fixed-size row per vessel containing normalized position,
  speed, heading sine/cosine, timestamp, active flag, and assignment flag.
- `nearest_distance`: normalized nearest-neighbour distance for each vessel.
- `operator`: normalized budget, operator status, preparation/assistance
  timers, elapsed assistance, and simulation time.

The allocation action is discrete: `0` means idle and `1..N` selects a vessel.
The controller action sent to `FleetSimulator` is continuous:
`[dx, dy, d_heading]` for each active vessel.

The reward is calculated from events that begin during the current step:

```text
reward = wg * newly_completed_journeys
       - wc * newly_started_collisions
       - wm * newly_started_near_misses
       - wh * accepted_interventions
```

This separation lets the allocation policy learn *when and whom to assist*
without coupling it to the BC navigation model or the simulator's per-ship
kinematics.
