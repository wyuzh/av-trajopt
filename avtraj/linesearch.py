"""Filter line search (Fletcher & Leyffer 2002; Waechter & Biegler 2006).

A merit function  phi + nu * theta  needs a well-chosen penalty nu and, on
curved constraints, rejects full Newton steps that slightly increase theta
(the Maratos effect).  A filter instead treats the two goals separately:

    theta - constraint violation,   phi - objective (or barrier objective).

A trial point is acceptable if it is not dominated by any filter entry and it
improves theta or phi by a small margin.  When the iterate is nearly feasible
and the step is a descent direction for phi, an Armijo decrease of phi is
required instead ("f-type" step); otherwise the current point is added to the
filter ("h-type" step).
"""
from __future__ import annotations


class Filter:
    def __init__(self, theta0, gamma_theta=1e-5, gamma_phi=1e-5, eta=1e-4,
                 s_theta=1.1, s_phi=2.3):
        self.th_max = 1e4 * max(1.0, theta0)
        self.th_min = 1e-4 * max(1.0, theta0)
        self.g_th, self.g_ph, self.eta = gamma_theta, gamma_phi, eta
        self.s_th, self.s_ph = s_theta, s_phi
        self.entries = []

    def reset(self):
        self.entries = []

    def add(self, th, ph):
        self.entries.append(((1 - self.g_th) * th, ph - self.g_ph * th))

    def dominated(self, th_t, ph_t):
        """True if (th_t, ph_t) is unacceptable to the filter."""
        return th_t > self.th_max or any(th_t >= tf and ph_t >= pf for tf, pf in self.entries)

    def accept(self, th_t, ph_t, th, ph, m, alpha):
        """Is the trial (th_t, ph_t) acceptable from (th, ph) with step alpha and
        directional derivative m of phi?  Updates the filter on h-type steps."""
        if self.dominated(th_t, ph_t):
            return False
        switching = m < 0 and alpha * (-m) ** self.s_ph > th ** self.s_th
        if th <= self.th_min and switching:
            return ph_t <= ph + self.eta * alpha * m
        if th_t <= (1 - self.g_th) * th or ph_t <= ph - self.g_ph * th:
            self.add(th, ph)
            return True
        return False
