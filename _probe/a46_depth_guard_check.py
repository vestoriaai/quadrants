"""A46 depth-cap guard verification: runaway runtime recursion must fail fast
with the named error (no OOM), static recursion must pass."""
import resource
import sys

import quadrants as qd

qd.init(arch=qd.cpu, offline_cache=False, cfg_optimization=False)

# 1) static-terminated recursion: legal, must compute
@qd.func
def fib_bar(x: qd.template()) -> qd.types.matrix(2, 2, qd.i32):
  if qd.static(x):
    mat = fib_bar(x // 2)
    mat = mat @ mat
    if qd.static(x % 2):
      mat = mat @ qd.types.matrix(2, 2, qd.i32)([[1, 1], [1, 0]])
    return mat
  else:
    return qd.types.matrix(2, 2, qd.i32)([[1, 0], [0, 1]])


@qd.kernel
def fib_k(x: qd.template()) -> qd.i32:
  return fib_bar(x)[1, 0]


fib_k(5)
qd.sync()
v = fib_k(9)
qd.sync()
print('static recursion fib(9)[1,0] =', v, '(expect 34)')
assert v == 34, v

# 2) runaway runtime recursion: must raise the named error quickly
from quadrants.lang.exception import QuadrantsSyntaxError


@qd.func
def runaway(x: qd.i32) -> qd.i32:
  return runaway(x + 1)


@qd.kernel
def run_k(x: qd.i32) -> qd.i32:
  return runaway(x)


try:
  run_k(0)
  qd.sync()
  print('ERROR: runaway recursion did not raise')
  sys.exit(1)
except Exception as e:  # may surface wrapped at the kernel boundary
  assert 'inline cycle' in str(e) or 'inline depth' in str(e), f'{type(e)}: {e}'
  print('runaway recursion raised named error: OK')

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f'peak RSS {rss:.0f} MB (guard held)')
print('ALL OK')
