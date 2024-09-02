"""MinAtar/Freeway: A fork of github.com/kenjyoung/MinAtar

The authors of original MinAtar implementation are:
    * Kenny Young (kjyoung@ualberta.ca)
    * Tian Tian (ttian@ualberta.ca)
The original MinAtar implementation is distributed under GNU General Public License v3.0
    * https://github.com/kenjyoung/MinAtar/blob/master/License.txt
"""
from typing import Literal, Optional

import jax
from jax import numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

player_speed = jnp.array(3, dtype=jnp.int32)
time_limit = jnp.array(2500, dtype=jnp.int32)

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
ZERO = jnp.array(0, dtype=jnp.int32)
ONE = jnp.array(1, dtype=jnp.int32)
NINE = jnp.array(9, dtype=jnp.int32)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((10, 10, 7), dtype=jnp.bool_)
    rewards: Array = jnp.zeros(1, dtype=jnp.float32)  # (1,)
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(3, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    # --- MinAtar Freeway specific ---
    _cars: Array = jnp.zeros((8, 4), dtype=jnp.int32)
    _pos: Array = jnp.array(9, dtype=jnp.int32)
    _move_timer: Array = jnp.array(player_speed, dtype=jnp.int32)
    _terminate_timer: Array = jnp.array(time_limit, dtype=jnp.int32)
    _terminal: Array = jnp.array(False, dtype=jnp.bool_)
    _last_action: Array = jnp.array(0, dtype=jnp.int32)

    @property
    def env_id(self) -> core.EnvId:
        return "minatar-freeway"

    def to_svg(
        self,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
    ) -> str:
        del color_theme, scale
        from .utils import visualize_minatar

        return visualize_minatar(self)

    def save_svg(
        self,
        filename,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
    ) -> None:
        from .utils import visualize_minatar

        visualize_minatar(self, filename)


class MinAtarFreeway(core.Env):
    def __init__(
        self,
        *,
        use_minimal_action_set: bool = True,
        sticky_action_prob: float = 0.1,
    ):
        super().__init__()
        self.use_minimal_action_set = use_minimal_action_set
        self.sticky_action_prob: float = sticky_action_prob
        self.minimal_action_set = jnp.int32([0, 2, 4])
        self.legal_action_mask = jnp.ones(6, dtype=jnp.bool_)
        if self.use_minimal_action_set:
            self.legal_action_mask = jnp.ones(
                self.minimal_action_set.shape[0], dtype=jnp.bool_
            )

    def step(
        self, state: core.State, action: Array, key: Optional[Array] = None
    ) -> core.State:
        assert key is not None, (
            "v2.0.0 changes the signature of step. Please specify PRNGKey at the third argument:\n\n"
            "  * <  v2.0.0: step(state, action)\n"
            "  * >= v2.0.0: step(state, action, key)\n\n"
            "See v2.0.0 release note for more details:\n\n"
            "  https://github.com/sotetsuk/pgx/releases/tag/v2.0.0"
        )
        return super().step(state, action, key)

    def _init(self, key: PRNGKey) -> State:
        state = _init(rng=key)  # type: ignore
        state = state.replace(legal_action_mask=self.legal_action_mask)  # type: ignore
        return state  # type: ignore

    def _step(self, state: core.State, action, key) -> State:
        assert isinstance(state, State)
        state = state.replace(legal_action_mask=self.legal_action_mask)  # type: ignore
        action = jax.lax.select(
            self.use_minimal_action_set,
            self.minimal_action_set[action],
            action,
        )
        return _step(state, action, key, self.sticky_action_prob)  # type: ignore

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state)

    @property
    def id(self) -> core.EnvId:
        return "minatar-freeway"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def num_players(self):
        return 1


def _step(
    state: State,
    action: Array,
    key,
    sticky_action_prob,
):
    action = jnp.int32(action)
    key0, key1 = jax.random.split(key, 2)
    action = jax.lax.cond(
        jax.random.uniform(key0) < sticky_action_prob,
        lambda: state._last_action,
        lambda: action,
    )
    speeds, directions = _random_speed_directions(key1)
    return _step_det(state, action, speeds=speeds, directions=directions)


def _init(rng: Array) -> State:
    speeds, directions = _random_speed_directions(rng)
    return _init_det(speeds=speeds, directions=directions)


def _step_det(
    state: State,
    action: Array,
    speeds: Array,
    directions: Array,
):
    cars = state._cars
    pos = state._pos
    move_timer = state._move_timer
    terminate_timer = state._terminate_timer
    terminal = state._terminal
    last_action = action

    r = jnp.array(0, dtype=jnp.float32)

    move_timer, pos = jax.lax.cond(
        (action == 2) & (move_timer == 0),
        lambda: (player_speed, jax.lax.max(ZERO, pos - ONE)),
        lambda: (move_timer, pos),
    )
    move_timer, pos = jax.lax.cond(
        (action == 4) & (move_timer == 0),
        lambda: (player_speed, jax.lax.min(NINE, pos + ONE)),
        lambda: (move_timer, pos),
    )

    # Win condition
    cars, r, pos = jax.lax.cond(
        pos == 0,
        lambda: (
            _randomize_cars(speeds, directions, cars, initialize=False),
            r + 1,
            NINE,
        ),
        lambda: (cars, r, pos),
    )

    pos, cars = _update_cars(pos, cars)

    # Update various timers
    move_timer = jax.lax.cond(
        move_timer > 0, lambda: move_timer - 1, lambda: move_timer
    )
    terminate_timer -= ONE
    terminal = terminate_timer < 0

    next_state = state.replace(  # type: ignore
        _cars=cars,
        _pos=pos,
        _move_timer=move_timer,
        _terminate_timer=terminate_timer,
        _terminal=terminal,
        _last_action=last_action,
        rewards=r[jnp.newaxis],
        terminated=terminal,
    )

    return next_state


def _update_cars(pos, cars):
    def _update_stopped_car(pos, car):
        car = car.at[2].set(jax.lax.abs(car[3]))
        car = jax.lax.cond(
            car[3] > 0, lambda: car.at[0].add(1), lambda: car.at[0].add(-1)
        )
        car = jax.lax.cond(car[0] < 0, lambda: car.at[0].set(9), lambda: car)
        car = jax.lax.cond(car[0] > 9, lambda: car.at[0].set(0), lambda: car)
        pos = jax.lax.cond(
            (car[0] == 4) & (car[1] == pos), lambda: NINE, lambda: pos
        )
        return pos, car

    def _update_car(pos, car):
        pos = jax.lax.cond(
            (car[0] == 4) & (car[1] == pos), lambda: NINE, lambda: pos
        )
        pos, car = jax.lax.cond(
            car[2] == 0,
            lambda: _update_stopped_car(pos, car),
            lambda: (pos, car.at[2].add(-1)),
        )
        return pos, car

    pos, cars = jax.lax.scan(_update_car, pos, cars)

    return pos, cars


def _init_det(speeds: Array, directions: Array) -> State:
    cars = _randomize_cars(speeds, directions, initialize=True)
    return State(_cars=cars)  # type: ignore


def _randomize_cars(
    speeds: Array,
    directions: Array,
    cars: Array = jnp.zeros((8, 4), dtype=int),
    initialize: bool = False,
) -> Array:
    speeds *= directions

    def _init(_cars):
        _cars = _cars.at[:, 1].set(jnp.arange(1, 9))
        _cars = _cars.at[:, 2].set(jax.lax.abs(speeds))
        _cars = _cars.at[:, 3].set(speeds)
        return _cars

    def _update(_cars):
        _cars = _cars.at[:, 2].set(abs(speeds))
        _cars = _cars.at[:, 3].set(speeds)
        return _cars

    return jax.lax.cond(initialize, _init, _update, cars)


def _random_speed_directions(rng):
    rng1, rng2 = jax.random.split(rng, 2)
    speeds = jax.random.randint(rng1, [8], 1, 6, dtype=jnp.int32)
    directions = jax.random.choice(
        rng2, jnp.array([-1, 1], dtype=jnp.int32), [8]
    )
    return speeds, directions


def _observe(state: State) -> Array:
    obs = jnp.zeros((10, 10, 7), dtype=jnp.bool_)
    obs = obs.at[state._pos, 4, 0].set(TRUE)

    def _update_obs(i, _obs):
        car = state._cars[i]
        _obs = _obs.at[car[1], car[0], 1].set(TRUE)
        back_x = jax.lax.cond(
            car[3] > 0, lambda: car[0] - 1, lambda: car[0] + 1
        )
        back_x = jax.lax.cond(back_x < 0, lambda: NINE, lambda: back_x)
        back_x = jax.lax.cond(back_x > 9, lambda: ZERO, lambda: back_x)
        trail = jax.lax.abs(car[3]) + 1
        _obs = _obs.at[car[1], back_x, trail].set(TRUE)
        return _obs

    obs = jax.lax.fori_loop(0, 8, _update_obs, obs)
    return obs
