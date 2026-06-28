# Operator And Skill Inventories

## Purpose

Build and evaluate reusable banks of operators, typed primitives, verified skills, and composable transformations that small models can search, shortlist, or call.

## Why This Is A Program

Operator coverage and shortlisting are not just implementation details; they define what a small model can express. This program studies inventory growth, disambiguation, scaling, and transfer.

## Progress Signals

- Larger inventories improve held-out target coverage without collapsing selection precision.
- Shortlisters reduce search cost while preserving oracle coverage.
- Active disambiguation resolves type-colliding primitives.
- Inventory entries carry enough metadata to be reused safely.

## Boundaries

This program owns the bank. Evidence gathering and final selection are separate programs when the bank is already producing candidates.
