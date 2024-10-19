from typing import TypeVar, Annotated

from typer import Argument as TyperArgument, Option as TyperOption


T = TypeVar("T")


def argument(t: T, h: str):
    return Annotated[t, TyperArgument(help=h)]


def option(t: T, h: str):
    return Annotated[t, TyperOption(help=h)]
