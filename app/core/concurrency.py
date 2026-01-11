import asyncio
import functools
from typing import Any, Callable, TypeVar

T = TypeVar("T")

async def run_in_threadpool(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Run a synchronous function in a separate thread to avoid blocking the main event loop.
    Useful for CPU-bound operations like Orange3 data loading.
    """
    loop = asyncio.get_running_loop()
    if kwargs:
        func = functools.partial(func, **kwargs)
    return await loop.run_in_executor(None, func, *args)
