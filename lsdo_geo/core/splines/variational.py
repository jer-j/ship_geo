"""Shared CSDL assembly for constrained variational geometry problems."""

from __future__ import annotations

from dataclasses import dataclass

import csdl_alpha as csdl
import numpy as np


@dataclass(frozen=True)
class ConstraintHandle:
    """Location of a local residual within the global constraint vector."""

    start: int
    stop: int


@dataclass
class VariationalResult:
    """CSDL variables produced by one global KKT solve."""

    objective: csdl.Variable
    stationarity_residuals: tuple[csdl.Variable, ...]
    constraint_residual: csdl.Variable | None
    lagrange_multipliers: csdl.Variable | None

    @property
    def stationarity_residual(self) -> csdl.Variable:
        """Return all stationarity residuals as one vector."""
        flattened = tuple(
            residual.flatten() for residual in self.stationarity_residuals
        )
        if len(flattened) == 1:
            return flattened[0]
        return csdl.concatenate(flattened)


class VariationalSystem:
    """Assemble many geometry states into one CSDL KKT/Newton solve.

    Geometry primitives add implicit coefficient states, objective terms, and
    equality residuals. The system forms a single Lagrangian and calls one
    ``csdl_alpha.nonlinear_solvers.Newton`` instance. Cross-primitive
    constraints can therefore be added before the solve without nesting or
    sequencing nonlinear solvers.
    """

    def __init__(self, name: str = "geometry") -> None:
        self.name = name
        self.states: list[csdl.ImplicitVariable] = []
        self.state_names: list[str] = []
        self.objective_terms: list[csdl.Variable] = []
        self.constraint_terms: list[csdl.Variable] = []
        self._num_constraints = 0
        self._solved = False

    def add_state(self, state: csdl.ImplicitVariable, name: str | None = None) -> int:
        """Register an implicit geometry state and return its stable index."""
        if self._solved:
            raise RuntimeError("cannot add a state after the system has been solved.")
        if not isinstance(state, csdl.ImplicitVariable):
            raise TypeError("variational states must be csdl.ImplicitVariable objects.")
        self.states.append(state)
        self.state_names.append(name or f"state_{len(self.states) - 1}")
        return len(self.states) - 1

    def add_objective(self, term: csdl.Variable) -> None:
        """Add a scalar contribution to the global variational objective."""
        if self._solved:
            raise RuntimeError("cannot add an objective after the solve.")
        if not isinstance(term, csdl.Variable) or term.size != 1:
            raise ValueError("each objective term must be a scalar CSDL variable.")
        self.objective_terms.append(term)

    def add_constraint(self, residual: csdl.Variable) -> ConstraintHandle:
        """Add an equality residual and return its global vector location."""
        if self._solved:
            raise RuntimeError("cannot add a constraint after the solve.")
        if not isinstance(residual, csdl.Variable):
            raise TypeError("constraint residuals must be CSDL variables.")
        flattened = residual.flatten()
        handle = ConstraintHandle(
            start=self._num_constraints,
            stop=self._num_constraints + int(flattened.size),
        )
        self.constraint_terms.append(flattened)
        self._num_constraints = handle.stop
        return handle

    def solve(
        self,
        tolerance: float = 1.0e-10,
        max_iter: int = 100,
        print_status: bool = False,
    ) -> VariationalResult:
        """Form and solve the global KKT system exactly once."""
        if self._solved:
            raise RuntimeError("a VariationalSystem can only be solved once.")
        try:
            csdl.get_current_recorder()
        except ValueError as error:
            raise RuntimeError(
                "VariationalSystem.solve requires an active csdl.Recorder."
            ) from error
        if not self.states:
            raise ValueError("at least one implicit geometry state is required.")
        if not self.objective_terms:
            raise ValueError("at least one variational objective term is required.")

        objective = self.objective_terms[0]
        for term in self.objective_terms[1:]:
            objective = objective + term

        constraint_residual: csdl.Variable | None = None
        multipliers: csdl.Variable | None = None
        lagrangian = objective
        if self.constraint_terms:
            if len(self.constraint_terms) == 1:
                constraint_residual = self.constraint_terms[0]
            else:
                constraint_residual = csdl.concatenate(tuple(self.constraint_terms))
            multipliers = csdl.ImplicitVariable(
                value=np.zeros(self._num_constraints),
                name=f"{self.name}_lagrange_multipliers",
            )
            lagrangian = lagrangian + csdl.vdot(multipliers, constraint_residual)

        stationarity = tuple(
            csdl.derivative(lagrangian, state).reshape(state.shape)
            for state in self.states
        )
        solver = csdl.nonlinear_solvers.Newton(
            name=f"{self.name}_newton",
            tolerance=tolerance,
            max_iter=max_iter,
            print_status=print_status,
            residual_jac_kwargs={"concatenate_ofs": True},
        )
        for state, residual in zip(self.states, stationarity):
            solver.add_state(state, residual)
        if multipliers is not None and constraint_residual is not None:
            solver.add_state(multipliers, constraint_residual)
        solver.run()

        self._solved = True
        return VariationalResult(
            objective=objective,
            stationarity_residuals=stationarity,
            constraint_residual=constraint_residual,
            lagrange_multipliers=multipliers,
        )
