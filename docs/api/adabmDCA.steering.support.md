# Steering constraints

## `HammingWindow`

`HammingWindow(reference, minimum, maximum)` restricts the number of
substitutions to an inclusive interval. `reference` is an aligned
model-length string in the model alphabet. Supply multiple windows in
`SteeringConfig.windows` to take their intersection.

## `HammingBias`

`HammingBias(reference, weight)` adds `weight * HammingDistance` to the
reduced energy. It is represented as a per-site field offset. Set a positive
weight to favor proximity or a negative weight to favor substitutions.

See the [steering API](adabmDCA.steering.sampling.md) and the
[Python examples](../usage/steering.md).
