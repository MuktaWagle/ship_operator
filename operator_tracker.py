"""Operator state and budget tracker for autonomous vessel remote assistance

- Tracks operator status: Available, Preparing, Assisting (h_t)
- Tracks remaining assistance budget (b_t) and elapsed simulation time (T_t)
- Implements assignment of assistance for fixed duration tau with buffer delay delta
- Enforces single-operator constraint (at most one vessel assisted concurrently)
- Enforces total active assistance budget limit B
- Provides action-masking modules for RL agents (DQN / baselines)
- Emits initiated intervention indicators (I_t) for reward calculation
- Produces normalized and structured observation vectors for RL integration
"""

from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np


class OperatorStatus(IntEnum):
    """Operator operational status h_t

    - AVAILABLE (0): Operator is idle and ready to take an assignment
    - PREPARING (1): Operator is in buffer delay delta (communication/setup delay)
    - ASSISTING (2): Operator is actively assisting the assigned vessel
    """
    AVAILABLE = 0
    PREPARING = 1
    ASSISTING = 2

    @property
    def label(self) -> str:
        return self.name.capitalize()


class OperatorBudgetTracker:
    """Manages the remote operator's state, queue, and intervention budget

    State representation:
        s_t = (..., b_t, h_t, T_t)
        - b_t: remaining assistance budget (seconds)
        - h_t: operator status (Available=0, Preparing=1, Assisting=2)
        - T_t: total elapsed simulation time (seconds)

    Action space:
        a_t in {0, 1, ..., N}
        - a_t = 0: No assignment (no-op / preserve operator availability)
        - a_t = i: Assign assistance to vessel i for fixed duration tau
          Assistance activates after communication buffer delta
    """

    def __init__(
        self,
        vessel_ids: List[int],
        total_budget: float = 300.0,
        duration: float = 60.0,
        delay: float = 10.0,
        dt: float = 10.0,
        max_simulation_time: float = 10000.0,
        allow_partial_duration: bool = False,
    ):
        """Initialize the operator state and budget tracker.

        Args:
            vessel_ids: Ordered list of unique vessel IDs in the fleet
            total_budget: Total assistance budget B in simulated seconds (default 300s)
            duration: Fixed intervention duration tau in simulated seconds (default 60s)
            delay: Communication / buffer delay delta in simulated seconds (default 10s)
            dt: Simulation time step in seconds (default 10s, matching FleetSimulator)
            max_simulation_time: Max simulation duration for normalizing T_t (default 10000s)
            allow_partial_duration: If True, allows an intervention if remaining budget > 0
                even if less than tau. If False (default), requires remaining budget >= tau
        """
        if not vessel_ids:
            raise ValueError("Fleet must contain at least one vessel ID.")

        if len(set(vessel_ids)) != len(vessel_ids):
            raise ValueError("Vessel IDs must be unique.")

        if dt <= 0:
            raise ValueError("dt must be positive.")

        if total_budget < 0:
            raise ValueError("total_budget must be non-negative.")

        if duration <= 0:
            raise ValueError("duration (tau) must be positive.")

        if delay < 0:
            raise ValueError("delay (delta) must be non-negative.")

        if not np.isclose(duration / dt, round(duration / dt)):
            raise ValueError("duration must be an integer multiple of dt.")

        if not np.isclose(delay / dt, round(delay / dt)):
            raise ValueError("delay must be an integer multiple of dt.")

        self.vessel_ids: List[int] = list(vessel_ids)
        self.num_vessels: int = len(self.vessel_ids)
        self.total_budget: float = float(total_budget)
        self.duration: float = float(duration)
        self.delay: float = float(delay)
        self.dt: float = float(dt)
        self.max_simulation_time: float = float(max_simulation_time)
        self.allow_partial_duration: bool = bool(allow_partial_duration)

        # Mapping between action index (1..N) and vessel ID
        self._action_to_ship: Dict[int, int] = {
            i + 1: ship_id for i, ship_id in enumerate(self.vessel_ids)
        }
        self._ship_to_action: Dict[int, int] = {
            ship_id: i + 1 for i, ship_id in enumerate(self.vessel_ids)
        }

        # Dynamic state variables initialized by reset()
        self.remaining_budget: float = self.total_budget
        self.status: OperatorStatus = OperatorStatus.AVAILABLE
        self.assigned_ship_id: Optional[int] = None
        self.preparation_timer: float = 0.0
        self.assistance_timer: float = 0.0
        self.simulation_time: float = 0.0
        self.step_count: int = 0

        # Statistics & step signals
        self.interventions_initiated_this_step: int = 0
        self.total_interventions: int = 0
        self.total_active_assistance_time: float = 0.0
        self.history: List[Dict[str, Any]] = []

        self.reset()

    def reset(self) -> Dict[str, Any]:
        """Reset the operator state and budget tracker for a new episode.

        Returns:
            Dictionary snapshot of the initial operator state.
        """
        self.remaining_budget = self.total_budget
        self.status = OperatorStatus.AVAILABLE
        self.assigned_ship_id = None
        self.preparation_timer = 0.0
        self.assistance_timer = 0.0
        self.simulation_time = 0.0
        self.step_count = 0
        self.interventions_initiated_this_step = 0
        self.total_interventions = 0
        self.total_active_assistance_time = 0.0
        self.history.clear()

        return self.get_state()

    @property
    def is_available(self) -> bool:
        """Return True if operator is free to accept a new assignment."""
        return self.status == OperatorStatus.AVAILABLE

    @property
    def is_preparing(self) -> bool:
        """Return True if operator is currently in the communication buffer delay."""
        return self.status == OperatorStatus.PREPARING

    @property
    def is_assisting(self) -> bool:
        """Return True if operator is actively controlling/assisting a vessel."""
        return self.status == OperatorStatus.ASSISTING

    @property
    def has_sufficient_budget(self) -> bool:
        """Return True if enough budget remains to initiate an intervention."""
        if self.allow_partial_duration:
            return self.remaining_budget > 0
        return self.remaining_budget >= self.duration

    def action_to_ship_id(self, action: int) -> Optional[int]:
        """Convert a discrete action index a_t in {0, ..., N} to vessel ID

        Args:
            action: 0 for no assignment, or 1..N for vessel index

        Returns:
            Vessel ID, or None if action == 0
        """
        if action == 0:
            return None
        if action not in self._action_to_ship:
            raise ValueError(
                f"Action {action} is invalid. Valid action range is 0 to {self.num_vessels}."
            )
        return self._action_to_ship[action]

    def ship_id_to_action(self, ship_id: Optional[int]) -> int:
        """Convert a vessel ID to discrete action index a_t in {0, ..., N}

        Args:
            ship_id: Vessel ID, or None for action 0

        Returns:
            Integer action index
        """
        if ship_id is None:
            return 0
        if ship_id not in self._ship_to_action:
            raise ValueError(f"Ship ID {ship_id} is not in fleet vessel IDs.")
        return self._ship_to_action[ship_id]

    def get_action_mask(
        self, active_ship_ids: Optional[Union[List[int], set]] = None
    ) -> np.ndarray:
        """Compute the boolean action mask for discrete actions {0, 1, ..., N}

        - Action 0 (no assignment) is ALWAYS valid (True)
        - Action i (vessel i) is valid (True) IF AND ONLY IF:
            1. Operator is Available (h_t == AVAILABLE)
            2. Remaining budget is sufficient (b_t >= tau, or > 0 if allow_partial)
            3. Vessel i is currently active in the fleet (if active_ship_ids provided)

        Args:
            active_ship_ids: Iterable of currently active vessel IDs in the fleet.
                If None, all fleet vessels are assumed active.

        Returns:
            np.ndarray of shape (N + 1,) and dtype bool, where mask[a] == True
            indicates that action a is legal.
        """
        mask = np.zeros(self.num_vessels + 1, dtype=bool)
        mask[0] = True  # a_t = 0 (no assignment) is always allowed

        # If operator is unavailable or budget is insufficient, only action 0 is allowed.
        if not self.is_available or not self.has_sufficient_budget:
            return mask

        active_set = (
            set(active_ship_ids) if active_ship_ids is not None else set(self.vessel_ids)
        )

        for action_idx, ship_id in self._action_to_ship.items():
            if ship_id in active_set:
                mask[action_idx] = True

        return mask

    def is_action_valid(
        self,
        action: Union[int, None],
        active_ship_ids: Optional[Union[List[int], set]] = None,
    ) -> bool:
        """Check whether a specific action is currently valid

        Args:
            action: Discrete action index 0..N, or vessel ID, or None (0)
            active_ship_ids: Optional collection of active ship IDs

        Returns:
            bool: True if action is legal, False otherwise
        """
        # Resolve action index
        if action is None:
            action_idx = 0
        elif action in self._ship_to_action:
            action_idx = self._ship_to_action[action]
        elif 0 <= action <= self.num_vessels:
            action_idx = action
        else:
            return False

        mask = self.get_action_mask(active_ship_ids)
        return bool(mask[action_idx])

    def dispatch(
        self,
        action: Union[int, None],
        active_ship_ids: Optional[Union[List[int], set]] = None,
        strict: bool = True,
    ) -> bool:
        """Dispatch an intervention decision a_t

        Called at the beginning of a simulation step before advancing the environment

        Args:
            action: Discrete action index in {0, ..., N}, or vessel ID, or None (no-op)
            active_ship_ids: Optional collection of active vessel IDs in the fleet
            strict: If True, raises ValueError on invalid actions. If False, returns False

        Returns:
            bool: True if action was successfully processed, False if invalid and not strict
        """
        # Reset step initiation indicator
        self.interventions_initiated_this_step = 0

        # Resolve action to action index and target vessel
        if action is None or action == 0:
            action_idx = 0
            target_ship = None
        elif action in self._ship_to_action:
            action_idx = self._ship_to_action[action]
            target_ship = action
        elif 1 <= action <= self.num_vessels:
            action_idx = action
            target_ship = self._action_to_ship[action]
        else:
            if strict:
                raise ValueError(f"Invalid action {action}.")
            return False

        # Action 0: No assignment
        if action_idx == 0:
            return True

        # Check legality of dispatching to target_ship
        if not self.is_action_valid(action_idx, active_ship_ids):
            msg = (
                f"Cannot assign vessel {target_ship} (action {action_idx}): "
                f"operator status={self.status.name}, "
                f"remaining_budget={self.remaining_budget:.1f}s, "
                f"active_ships={list(active_ship_ids) if active_ship_ids is not None else 'any'}."
            )
            if strict:
                raise ValueError(msg)
            return False

        # Valid intervention assignment
        self.assigned_ship_id = target_ship
        self.interventions_initiated_this_step = 1
        self.total_interventions += 1

        if self.delay > 0:
            # Buffer delay delta before assistance activates
            self.status = OperatorStatus.PREPARING
            self.preparation_timer = self.delay
            self.assistance_timer = min(self.duration, self.remaining_budget)
        else:
            # Immediate activation if delay == 0
            self.status = OperatorStatus.ASSISTING
            self.preparation_timer = 0.0
            self.assistance_timer = min(self.duration, self.remaining_budget)

        self.history.append({
            "step": self.step_count,
            "time": self.simulation_time,
            "event": "DISPATCH",
            "ship_id": target_ship,
            "status": self.status.name,
            "prep_timer": self.preparation_timer,
            "assist_timer": self.assistance_timer,
            "budget": self.remaining_budget,
        })

        return True

    def step(
        self,
        active_ship_ids: Optional[Union[List[int], set]] = None,
        dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Advance the operator and budget tracker by one simulation step dt

        Updates timers, manages state transitions:
        - Preparing -> Assisting (after buffer delta expires)
        - Assisting -> Available (after duration tau expires or budget is exhausted)
        - Early termination -> Available (if assisted ship arrived or retired)

        Args:
            active_ship_ids: Iterable of vessel IDs currently active in the fleet
            dt: Simulation interval in seconds. If None, uses self.dt

        Returns:
            Dictionary with transition notifications and current operator state
        """
        step_dt = float(dt) if dt is not None else self.dt
        if step_dt <= 0:
            raise ValueError("step dt must be positive.")

        transition_event = None
        active_set = set(active_ship_ids) if active_ship_ids is not None else None

        # Check if currently assigned vessel retired/arrived early
        if self.assigned_ship_id is not None and active_set is not None:
            if self.assigned_ship_id not in active_set:
                transition_event = f"EARLY_RELEASE_RETIRED_{self.assigned_ship_id}"
                self._release_operator()

        # Advance state timers
        if self.status == OperatorStatus.PREPARING:
            self.preparation_timer = max(0.0, self.preparation_timer - step_dt)
            if self.preparation_timer <= 1e-9:
                # Communication buffer finished -> start active assistance
                self.status = OperatorStatus.ASSISTING
                self.preparation_timer = 0.0
                transition_event = f"ASSISTANCE_STARTED_{self.assigned_ship_id}"

        elif self.status == OperatorStatus.ASSISTING:
            # Consume budget and advance assistance timer
            actual_assist_dt = min(step_dt, self.assistance_timer, self.remaining_budget)
            self.remaining_budget = max(0.0, self.remaining_budget - actual_assist_dt)
            self.total_active_assistance_time += actual_assist_dt
            self.assistance_timer = max(0.0, self.assistance_timer - step_dt)

            if self.assistance_timer <= 1e-9 or self.remaining_budget <= 1e-9:
                reason = "DURATION_EXPIRED" if self.assistance_timer <= 1e-9 else "BUDGET_EXHAUSTED"
                transition_event = f"{reason}_{self.assigned_ship_id}"
                self._release_operator()

        # Capture and reset step intervention indicator for next step
        initiated_in_step = self.interventions_initiated_this_step
        self.interventions_initiated_this_step = 0

        # Advance simulation clock
        self.step_count += 1
        self.simulation_time += step_dt

        # Log transition if any occurred
        if transition_event:
            self.history.append({
                "step": self.step_count,
                "time": self.simulation_time,
                "event": transition_event,
                "status": self.status.name,
                "budget": self.remaining_budget,
            })

        state = self.get_state()
        state["transition_event"] = transition_event
        state["interventions_initiated"] = initiated_in_step
        return state

    def _release_operator(self) -> None:
        """Internal helper to free the operator back to AVAILABLE."""
        self.status = OperatorStatus.AVAILABLE
        self.assigned_ship_id = None
        self.preparation_timer = 0.0
        self.assistance_timer = 0.0

    def get_state(self) -> Dict[str, Any]:
        """Return an independent dictionary snapshot of the operator & budget state.

        Matches the state formulation (..., b_t, h_t, T_t):
        - b_t: budget
        - h_t: status
        - T_t: simulation_time
        """
        return {
            "budget": float(self.remaining_budget),
            "total_budget": float(self.total_budget),
            "budget_ratio": float(
                self.remaining_budget / self.total_budget if self.total_budget > 0 else 0.0
            ),
            "status": int(self.status.value),
            "status_name": str(self.status.name),
            "assigned_ship_id": self.assigned_ship_id,
            "prep_time_remaining": float(self.preparation_timer),
            "assist_time_remaining": float(self.assistance_timer),
            "simulation_time": float(self.simulation_time),
            "step": int(self.step_count),
            "interventions_initiated": int(self.interventions_initiated_this_step),
            "total_interventions": int(self.total_interventions),
            "total_active_time": float(self.total_active_assistance_time),
            "can_assign": bool(self.is_available and self.has_sufficient_budget),
        }

    def get_observation_vector(self, normalize: bool = True) -> np.ndarray:
        """Generate a 1D feature vector for RL observation concatenation

        Features:
        - Remaining budget b_t (normalized to [0, 1] if normalize=True)
        - Operator status h_t as one-hot: [is_available, is_preparing, is_assisting]
        - Preparation time remaining (normalized to [0, 1] by delta)
        - Assistance time remaining (normalized to [0, 1] by tau)
        - Total simulation time T_t (normalized to [0, 1] by max_simulation_time)

        Args:
            normalize: If True, scales continuous features to roughly [0, 1]

        Returns:
            np.ndarray of shape (7,) with dtype float32
        """
        if normalize:
            norm_budget = (
                self.remaining_budget / self.total_budget if self.total_budget > 0 else 0.0
            )
            norm_prep = (
                self.preparation_timer / self.delay if self.delay > 0 else 0.0
            )
            norm_assist = (
                self.assistance_timer / self.duration if self.duration > 0 else 0.0
            )
            norm_time = (
                min(1.0, self.simulation_time / self.max_simulation_time)
                if self.max_simulation_time > 0
                else 0.0
            )
        else:
            norm_budget = self.remaining_budget
            norm_prep = self.preparation_timer
            norm_assist = self.assistance_timer
            norm_time = self.simulation_time

        # One-hot encoding of status h_t (3 elements)
        status_one_hot = [
            1.0 if self.status == OperatorStatus.AVAILABLE else 0.0,
            1.0 if self.status == OperatorStatus.PREPARING else 0.0,
            1.0 if self.status == OperatorStatus.ASSISTING else 0.0,
        ]

        vec = [
            norm_budget,
            status_one_hot[0],
            status_one_hot[1],
            status_one_hot[2],
            norm_prep,
            norm_assist,
            norm_time,
        ]

        return np.array(vec, dtype=np.float32)

    def sync_with_fleet(self, fleet) -> None:
        """Optional synchronization helper with a FleetSimulator instance

        Updates simulation time and cleans up if assigned ship retired

        Args:
            fleet: An instance of FleetSimulator
        """
        active_ids = fleet.active_ship_ids
        if self.assigned_ship_id is not None and self.assigned_ship_id not in active_ids:
            self._release_operator()
