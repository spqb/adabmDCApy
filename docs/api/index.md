# Python API reference

These pages are generated from the current Python source with `lazydocs`. Begin with the [usage guides](../usage/index.md) for complete workflows; use these pages for callable signatures and result fields. The [symbol index](symbols.md) links to individual documented functions and classes.

| Workflow | Modules |
| --- | --- |
| Train and resume | [Training](adabmDCA.api.training.md), [PTT configuration](adabmDCA.ptt.config.md), [Training configuration](adabmDCA.training_config.md) |
| Generate and steer | [Sampling](adabmDCA.api.sampling.md), [PTT workflows](adabmDCA.api.ptt.md), [Steering](adabmDCA.steering.md), [PTT sampler](adabmDCA.ptt.sampler.md) |
| Work with results | [Model object](adabmDCA.api.model.md), [Result objects](adabmDCA.api.results.md) |
| Analyze | [Scoring](adabmDCA.api.scoring.md), [Contacts](adabmDCA.api.contacts.md), [Mutations](adabmDCA.api.mutations.md), [Entropy](adabmDCA.api.entropy.md) |
| Prepare data | [Input loading](adabmDCA.input_loading.md), [Splitting](adabmDCA.api.splitting.md), [Reintegration](adabmDCA.api.reintegration.md) |

For a new application, import high-level functions from `adabmDCA`, for example `from adabmDCA import PTTConfig, train_model, sample_sequences`.
