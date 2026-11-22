"""The task sentence. Recording, inspection and status all build it here, so the 12 strings in the dataset never drift."""

POCKETS = 6


def task(pocket, command):
    """Task text of an episode. The top camera has already inspected the tray; the policy is told where and to which bin."""
    return f"put the part from pocket {pocket} in the {command.upper()} bin"
