"""MinAtar/Breakout: A fork of github.com/kenjyoung/MinAtar

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

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
ZERO = jnp.array(0, dtype=jnp.int32)
ONE = jnp.array(1, dtype=jnp.int32)
TWO = jnp.array(2, dtype=jnp.int32)
THREE = jnp.array(3, dtype=jnp.int32)
FOUR = jnp.array(4, dtype=jnp.int32)
NINE = jnp.array(9, dtype=jnp.int32)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((10, 10, 4), dtype=jnp.bool_)
    rewards: Array = jnp.zeros(1, dtype=jnp.float32)  # (1,)
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(3, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    # --- MinAtar Breakout specific ---
    _ball_y: Array = THREE
    _ball_x: Array = ZERO
    _ball_dir: Array = TWO
    _pos: Array = FOUR
    _brick_map: Array = (
        jnp.zeros((10, 10), dtype=jnp.bool_).at[1:4, :].set(True)
    )
    _strike: Array = jnp.array(False, dtype=jnp.bool_)
    _last_x: Array = ZERO
    _last_y: Array = THREE
    _terminal: Array = jnp.array(False, dtype=jnp.bool_)
    _last_action: Array = ZERO

    @property
    def env_id(self) -> core.EnvId:
        return "minatar-breakout"

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


class MinAtarBreakout(core.Env):
    def __init__(
        self,
        *,
        use_minimal_action_set: bool = True,
        sticky_action_prob: float = 0.1,
    ):
        super().__init__()
        self.use_minimal_action_set = use_minimal_action_set
        self.sticky_action_prob: float = sticky_action_prob
        self.minimal_action_set = jnp.int32([0, 1, 3])
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
        return "minatar-breakout"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def num_players(self):
        return 1


def _step(
    state: State,
    action,
    key,
    sticky_action_prob,
):
    action = jnp.int32(action)
    action = jax.lax.cond(
        jax.random.uniform(key) < sticky_action_prob,
        lambda: state._last_action,
        lambda: action,
    )
    return _step_det(state, action)


def _init(rng: Array) -> State:
    ball_start = jax.random.choice(rng, 2)
    return _init_det(ball_start=ball_start)


def _step_det(state: State, action: Array):
    ball_y = state._ball_y
    ball_x = state._ball_x
    ball_dir = state._ball_dir
    pos = state._pos
    brick_map = state._brick_map
    strike = state._strike
    terminal = state._terminal

    r = jnp.array(0, dtype=jnp.float32)

    pos = _apply_action(pos, action)

    # Update ball position
    last_x = ball_x
    last_y = ball_y
    new_x, new_y = _update_ball_pos(ball_x, ball_y, ball_dir)

    new_x, ball_dir = jax.lax.cond(
        (new_x < 0) | (new_x > 9),
        lambda: _update_ball_pos_x(new_x, ball_dir),
        lambda: (new_x, ball_dir),
    )

    is_new_y_negative = new_y < 0
    is_strike = brick_map[new_y, new_x] == 1
    is_bottom = new_y == 9
    new_y, ball_dir = jax.lax.cond(
        is_new_y_negative,
        lambda: _update_ball_pos_y(ball_dir),
        lambda: (new_y, ball_dir),
    )
    strike_toggle = ~is_new_y_negative & is_strike
    r, strike, brick_map, new_y, ball_dir = jax.lax.cond(
        ~is_new_y_negative & is_strike & ~strike,
        lambda: _update_by_strike(
            r, brick_map, new_x, new_y, last_y, ball_dir, strike
        ),
        lambda: (r, strike, brick_map, new_y, ball_dir),
    )
    brick_map, new_y, ball_dir, terminal = jax.lax.cond(
        ~is_new_y_negative & ~is_strike & is_bottom,
        lambda: _update_by_bottom(
            brick_map, ball_x, new_x, new_y, pos, ball_dir, last_y, terminal
        ),
        lambda: (brick_map, new_y, ball_dir, terminal),
    )

    strike = jax.lax.cond(
        ~strike_toggle, lambda: jnp.zeros_like(strike), lambda: strike
    )

    state = state.replace(  # type: ignore
        _ball_y=new_y,
        _ball_x=new_x,
        _ball_dir=ball_dir,
        _pos=pos,
        _brick_map=brick_map,
        _strike=strike,
        _last_x=last_x,
        _last_y=last_y,
        _terminal=terminal,
        _last_action=action,
        rewards=r[jnp.newaxis],
        terminated=terminal,
    )
    return state


def _apply_action(pos, action):
    pos = jax.lax.cond(
        action == 1, lambda: jax.lax.max(ZERO, pos - ONE), lambda: pos
    )
    pos = jax.lax.cond(
        action == 3, lambda: jax.lax.min(NINE, pos + ONE), lambda: pos
    )
    return pos


def _update_ball_pos(ball_x, ball_y, ball_dir):
    return jax.lax.switch(
        ball_dir,
        [
            lambda: (ball_x - ONE, ball_y - ONE),
            lambda: (ball_x + ONE, ball_y - ONE),
            lambda: (ball_x + ONE, ball_y + ONE),
            lambda: (ball_x - ONE, ball_y + ONE),
        ],
    )


def _update_ball_pos_x(new_x, ball_dir):
    new_x = jax.lax.max(ZERO, new_x)
    new_x = jax.lax.min(NINE, new_x)
    ball_dir = jnp.array([1, 0, 3, 2], dtype=jnp.int32)[ball_dir]
    return new_x, ball_dir


def _update_ball_pos_y(ball_dir):
    ball_dir = jnp.array([3, 2, 1, 0], dtype=jnp.int32)[ball_dir]
    return ZERO, ball_dir


def _update_by_strike(r, brick_map, new_x, new_y, last_y, ball_dir, strike):
    brick_map = brick_map.at[new_y, new_x].set(False)
    new_y = last_y
    ball_dir = jnp.array([3, 2, 1, 0], dtype=jnp.int32)[ball_dir]
    return r + 1, jnp.ones_like(strike), brick_map, new_y, ball_dir


def _update_by_bottom(
    brick_map, ball_x, new_x, new_y, pos, ball_dir, last_y, terminal
):
    brick_map = jax.lax.cond(
        brick_map.sum() == 0,
        lambda: brick_map.at[1:4, :].set(True),
        lambda: brick_map,
    )
    new_y, ball_dir, terminal = jax.lax.cond(
        ball_x == pos,
        lambda: (
            last_y,
            jnp.array([3, 2, 1, 0], dtype=jnp.int32)[ball_dir],
            terminal,
        ),
        lambda: jax.lax.cond(
            new_x == pos,
            lambda: (
                last_y,
                jnp.array([2, 3, 0, 1], dtype=jnp.int32)[ball_dir],
                terminal,
            ),
            lambda: (new_y, ball_dir, jnp.array(True, dtype=jnp.bool_)),
        ),
    )
    return brick_map, new_y, ball_dir, terminal


def _init_det(ball_start: Array) -> State:
    ball_x, ball_dir = jax.lax.switch(
        ball_start,
        [lambda: (ZERO, TWO), lambda: (NINE, THREE)],
    )
    last_x = ball_x
    return State(
        _ball_x=ball_x, _ball_dir=ball_dir, _last_x=last_x
    )  # type: ignore


def _observe(state: State) -> Array:
    obs = jnp.zeros((10, 10, 4), dtype=jnp.bool_)
    obs = obs.at[state._ball_y, state._ball_x, 1].set(True)
    obs = obs.at[9, state._pos, 0].set(True)
    obs = obs.at[state._last_y, state._last_x, 2].set(True)
    obs = obs.at[:, :, 3].set(state._brick_map)
    return obs
