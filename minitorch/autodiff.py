from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_list = list(vals)

    vals_list[arg] += epsilon
    f_diff = f(*vals_list)

    vals_list[arg] -= 2 * epsilon
    b_diff = f(*vals_list)

    return (f_diff - b_diff) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    res = []
    visited = set()

    def iter_and_push(root: Variable):
        if root.is_constant() or root.unique_id in visited:
            return

        if not root.is_leaf():
            for node in root.parents:
                iter_and_push(node)

        res.append(root)
        visited.add(root.unique_id)

    iter_and_push(variable)
    return reversed(res)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    ordered_nodes = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}
    for node in ordered_nodes:
        d_out = derivatives[node.unique_id]
        if node.is_leaf():
            node.accumulate_derivative(d_out)
        else:
            d_inputs = node.chain_rule(d_out)
            for parent, d_input in d_inputs:
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = d_input
                else:
                    derivatives[parent.unique_id] += d_input


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
