"""Smooth zero-mean joint-space perturbations for exploratory data collection.

A successful expert trajectory is replayed with a perturbation added to the joint
commands over a window at the end of the episode. Widening the window from the same
seed fans one success out into a family of outcomes, some of which still succeed and
some of which do not -- which is the point: both are kept and labelled.

The perturbation is built from sine modes that complete a whole number of periods
across the window:

    c_j(t) = sum_k a_jk * sin(2*pi*k*t/T),    t in [0, T]

Every mode is exactly zero at both ends of the window and integrates to zero over it.
So the signal is continuous where the window opens, continuous where it closes, and
zero-mean by construction -- no taper, and no drift that would just amount to a
different final pose. Mode amplitudes fall off as 1/k, so the result is smooth rather
than jittery, and the whole thing is rescaled to a stated peak in radians.
"""

import numpy as np


class JointCurve:
    """Additive joint-space curve over a trailing window of an episode.

    Args:
        n_joints: number of joints the curve is applied to.
        num_steps: total control steps in the episode.
        window_steps: length of the trailing window, in control steps. Clipped to
            num_steps, so "the whole episode" is just a window as long as the episode.
        amplitude: peak absolute deviation, in radians.
        n_modes: number of sine modes per joint.
        seed: seed for the mode amplitudes.
    """

    def __init__(self, n_joints, num_steps, window_steps, amplitude, n_modes=4, seed=None):
        self.n_joints = int(n_joints)
        self.num_steps = int(num_steps)
        self.window_steps = int(min(max(window_steps, 0), num_steps))
        self.amplitude = float(amplitude)
        self.n_modes = int(n_modes)
        self.start_step = self.num_steps - self.window_steps

        rng = np.random.default_rng(seed)
        if self.window_steps <= 1 or self.amplitude == 0.0 or self.n_joints == 0:
            self._table = np.zeros((max(self.window_steps, 0), self.n_joints))
            return

        # t normalised to [0, 1] across the window; endpoints included so the curve is
        # exactly zero on the step the window opens and the step it closes.
        t = np.linspace(0.0, 1.0, self.window_steps)
        k = np.arange(1, self.n_modes + 1)
        basis = np.sin(2.0 * np.pi * np.outer(t, k))          # (window, modes)
        weights = rng.normal(size=(self.n_modes, self.n_joints)) / k[:, None]
        table = basis @ weights                                # (window, joints)

        peak = np.max(np.abs(table))
        if peak > 0:
            table *= self.amplitude / peak
        self._table = table

    def offset(self, control_idx):
        """Curve value at one control step. Zero outside the window."""
        if self.window_steps <= 1:
            return np.zeros(self.n_joints)
        idx = control_idx - self.start_step
        if idx < 0 or idx >= self.window_steps:
            return np.zeros(self.n_joints)
        return self._table[idx]

    def summary(self):
        return {
            "n_joints": self.n_joints,
            "num_steps": self.num_steps,
            "window_steps": self.window_steps,
            "start_step": self.start_step,
            "amplitude": self.amplitude,
            "n_modes": self.n_modes,
        }


def window_steps_for_seconds(seconds, control_hz):
    """Control steps in a window given in seconds."""
    return int(round(float(seconds) * float(control_hz)))


class DualArmCurve:
    """One JointCurve per arm, sharing a window but drawn independently.

    The window is measured against the whole episode, not against one motion segment:
    an episode is many take_dense_action() calls and the plan's windows ("last 5 s",
    "last 10 s", ... up to the whole episode) are trailing windows of the episode. So
    num_steps must be the episode's total control-step count, which is known from a
    prior un-perturbed replay of the same seed.
    """

    def __init__(self, left_dim, right_dim, num_steps, window_steps, amplitude,
                 n_modes=4, seed=None):
        left_seed = None if seed is None else int(seed) * 2
        right_seed = None if seed is None else int(seed) * 2 + 1
        self.left = JointCurve(left_dim, num_steps, window_steps, amplitude, n_modes, left_seed)
        self.right = JointCurve(right_dim, num_steps, window_steps, amplitude, n_modes, right_seed)
        self.num_steps = int(num_steps)
        self.window_steps = self.left.window_steps
        self.amplitude = float(amplitude)
        self.seed = seed

    def offset(self, control_idx):
        """(left_offset, right_offset) at one absolute control step of the episode."""
        return self.left.offset(control_idx), self.right.offset(control_idx)

    def summary(self):
        return {
            "num_steps": self.num_steps,
            "window_steps": self.window_steps,
            "start_step": self.left.start_step,
            "amplitude": self.amplitude,
            "n_modes": self.left.n_modes,
            "seed": self.seed,
        }
