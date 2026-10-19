# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.server.exceptions import RequestRejectedError, ResourceNotFoundError


class TrainingExpiredError(ResourceNotFoundError):
    pass


class ProtocolMismatchError(RequestRejectedError):
    pass
