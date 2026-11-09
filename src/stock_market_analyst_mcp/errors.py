"""Domain errors. Their messages are meant to be read by the model and the end user."""


class AnalystError(ValueError):
    """Base class for anticipated, user-facing errors."""


class InvalidInputError(AnalystError):
    """An argument is malformed or out of range."""


class DataUnavailableError(AnalystError):
    """The data provider returned nothing usable for the request."""
