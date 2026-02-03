# Mutate Architectures (Genetic Algorithm)

Creates mutations of parent architectures by randomly perturbing
parameters. Used for genetic algorithm-style architecture search where
good-performing architectures are mutated to create the next generation.

## Usage

``` r
mutate_architectures(parents, n, mutation_rate = 0.3, config = NULL)
```

## Arguments

- parents:

  Data frame of parent architectures to mutate

- n:

  Number of mutated children to generate

- mutation_rate:

  Probability of mutating each parameter (default 0.3)

- config:

  Optional list specifying architecture parameter ranges (same format as
  `generate_architectures`). Used to bound mutations.

## Value

A data frame with mutated architecture specifications

## Details

Mutations include:

- depth: +1 or -1 (bounded by config)

- width: multiply by 2 or divide by 2 (bounded by config)

- activation: randomly swap to a different function from config

- reg/dropout: small random perturbation (bounded by config)
