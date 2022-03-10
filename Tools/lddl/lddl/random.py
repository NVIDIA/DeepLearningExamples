import random


def _swap_rng_state(new_state):
  old_state = random.getstate()
  random.setstate(new_state)
  return old_state


def randrange(stop, rng_state=None):
  orig_rng_state = _swap_rng_state(rng_state)
  n = random.randrange(stop)
  return n, _swap_rng_state(orig_rng_state)


def shuffle(x, rng_state=None):
  orig_rng_state = _swap_rng_state(rng_state)
  random.shuffle(x)
  return _swap_rng_state(orig_rng_state)


def sample(population, k, rng_state=None):
  orig_rng_state = _swap_rng_state(rng_state)
  s = random.sample(population, k)
  return s, _swap_rng_state(orig_rng_state)


def choices(population, weights=None, cum_weights=None, k=1, rng_state=None):
  orig_rng_state = _swap_rng_state(rng_state)
  c = random.choices(population, weights=weights, cum_weights=cum_weights, k=k)
  return c, _swap_rng_state(orig_rng_state)
