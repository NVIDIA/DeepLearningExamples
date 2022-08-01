class PerfAnalyzerException(Exception):
    def __init__(self, message: str):
        self._message = message

    def __str__(self):
        """
        Get the exception string representation.

        Returns
        -------
        str
            The message associated with this exception, or None if no message.
        """
        return self._message

    @property
    def message(self):
        """
        Get the exception message.

        Returns
        -------
        str
            The message associated with this exception, or None if no message.
        """
        return self._message
