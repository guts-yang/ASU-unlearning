"""RecodeLoop error types."""


class RecodeLoopError(RuntimeError):
    """Base error for RecodeLoop."""


class AssetBlocked(RecodeLoopError):
    """Raised when G0 does not pass and scientific evaluation must not run."""


class ConfirmationSealed(RecodeLoopError):
    """Raised on any access attempt to the sealed confirmation split."""


class ContractViolation(RecodeLoopError):
    """Raised when a dataclass/protocol invariant is broken."""
