"""Configured NTT schedule executors.

Each executor stores algorithm-specific schedule tables and RNS parameters,
then invokes the matching native NTT schemas. ``NativeNttImplementation`` in
``fhelium.backend.ntt.operations`` implements NTT IR operations and delegates
their transforms through one of these executors.
"""
