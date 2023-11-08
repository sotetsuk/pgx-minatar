import jax
import jax.numpy as jnp
import pgx
import pickle
from pgx_minatar.seaquest import MinAtarSeaquest


env = MinAtarSeaquest()

with open("prev_state.pkl", 'rb') as f:
    s = pickle.load(f)

s = s.replace(observation=env.observe(s, 0))
s.save_svg("prev.svg")

with open("expected_obs.pkl", 'rb') as f:
    o = pickle.load(f)

s = jax.jit(env.step)(s, jnp.int32(5))
s = s.replace(observation=env.observe(s, 0))
s.save_svg("next_pgx.svg")

# assert (s.observation == o).all()

# with open("error_state.pkl", 'rb') as f:
#     s = pickle.load(f)
#     print(s)
