import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize

class Dmp:
    def __init__(self, obs_param=None):
        self.a = 25.0                  # transformation gain
        self.b = 25.0 / 4.0
        self.as_ = 5.0                 # canonical decay
        self.ng = 20                  # basis functions

        self.iter = 100

        # canonical centers
        self.c = np.exp(-self.as_ * np.linspace(0, 1, self.ng))
        self.h = np.ones(self.ng) * (self.ng ** 1.5) / self.c

    # ==========================================================
    # Canonical System
    # ==========================================================
    def phase(self, t):
        tau = t[-1]
        dt = np.concatenate(([0], np.diff(t)))

        s = np.zeros(len(t))
        s[0] = 1.0

        for i in range(1, len(t)):
            s[i] = s[i - 1] - self.as_ * s[i - 1] * dt[i] / tau

        return s

    # ==========================================================
    # Basis Functions
    # ==========================================================
    def basis_functions(self, s):
        S = np.tile(s, (self.ng, 1))
        C = np.tile(self.c[:, None], (1, len(s)))

        psi = np.exp(-self.h[:, None] * (S - C) ** 2)
        return psi

    # ==========================================================
    # Imitation Learning
    # ==========================================================
    def imitate(self, x, dx, ddx, t, s, psi):

        tau = t[-1]
        g = x[-1, :]
        x0 = x[0, :]

        sigma = s

        ftarget = (tau**2 * ddx - self.a * (self.b * (g - x) - tau * dx))

        w = np.zeros((self.ng, x.shape[1]))

        for k in range(self.ng):

            numerator = ((psi[k, :] * sigma)[:, None] * ftarget).sum(axis=0)

            denominator = np.sum(psi[k, :] * sigma**2)

            w[k, :] = numerator / denominator

        return w, ftarget

    # ==========================================================
    # Generate Trajectory
    # ==========================================================
    def generate(self, w, x0, g, t, s, psi):

        tau = t[-1]
        dt = np.concatenate(([0], np.diff(t)))

        n = len(t)
        dim = len(x0)

        x = np.zeros((n, dim))
        dx = np.zeros((n, dim))
        ddx = np.zeros((n, dim))

        x[0, :] = x0

        for i in range(1, n):

            f = (psi[:, i] @ w) / np.sum(psi[:, i]) * s[i]

            ddx[i, :] = (self.a* (self.b * (g - x[i - 1, :])- tau * dx[i - 1, :])+ f) / tau**2

            dx[i, :] = dx[i - 1, :] + ddx[i, :] * dt[i]
            x[i, :] = x[i - 1, :] + dx[i, :] * dt[i]

        return x, dx, ddx

    # ==========================================================
    # PoWER RL Adaptation
    # ==========================================================
    def adapt(self, w, x0, g, t, s, psi, samples, elite_ratio):

        dim = w.shape[1]
        sigma = 0.1 * np.abs(w)

        best_reward = -np.inf

        reward_history = np.zeros(self.gen)

        x_best = None

        for it in range(self.gen):

            W = np.zeros((samples, self.ng, dim))
            R = np.zeros(samples)

            for k in range(samples):

                noise = sigma * np.random.randn(*w.shape)

                W[k] = w + noise

                x, _, _ = self.generate(W[k], x0, g, t, s, psi)

                R[k] = trajectory_reward(g, x, t, s)


            idx = np.argsort(R)[::-1]

            elite_count = round(elite_ratio * samples)

            elite = idx[:elite_count]

            w = np.mean(W[elite], axis=0)

            if R[idx[0]] > best_reward:

                best_reward = R[idx[0]]

                x_best, _, _ = self.generate(W[idx[0]], x0, g,t, s, psi,)

            reward_history[it] = np.max(R)
        
        
        stats = {"reward_history": reward_history}

        return x_best, w, stats

    # ==========================================================
    # PI² Adaptation
    # ==========================================================
    def adapt_pi2( self, w, x0, g, t, s, psi, samples ):

        dim = w.shape[1]
        sigma = 0.1 * np.abs(w)

        best_reward = -np.inf

        reward_history = np.zeros(self.gen)

        x_best = None

        for it in range(self.gen):

            W = np.zeros((samples, self.ng, dim))
            R = np.zeros(samples)

            for k in range(samples):

                noise = sigma * np.random.randn(*w.shape)

                W[k] = w + noise

                x, _, _ = self.generate( W[k], x0, g, t,s, psi)

                R[k] = trajectory_reward( g, x, t)

            S = -R

            S -= np.min(S)

            if np.max(S) > 0:
                S /= np.max(S)

            P = np.exp(-S)
            P /= np.sum(P)

            w_new = np.zeros_like(w)

            for k in range(samples):
                w_new += P[k] * W[k]

            w = w_new

            idx = np.argmax(R)

            if R[idx] > best_reward:

                best_reward = R[idx]

                x_best, _, _ = self.generate( W[idx], x0, g, t, s, psi)

            reward_history[it] = np.max(R)

        stats = {"reward_history": reward_history}

        return x_best, w, stats

# ==============================================================
# Reward Function
# ==============================================================

def trajectory_reward(target, x, time):

    w = 0.1
    thres = 0.01

    tau = time[-1]

    r = 0.0

    for i in range(len(time)):

        temp = target - x[i]

        if abs(time[i] - tau) < thres:
            r += w * np.exp(-np.linalg.norm(temp))
        else:
            r += ((1 - w)* np.exp(-np.linalg.norm(temp))/ tau)

    return r