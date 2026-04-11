"""Shared enums for the simulation core. No external dependencies."""
from enum import Enum


class RecsysType(str, Enum):
    TWHIN = "twhin-bert"
    TWITTER = "twitter"
    REDDIT = "reddit"
    RANDOM = "random"
