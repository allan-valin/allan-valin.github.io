# Thesis code: my part

From the bachelor thesis *Machine Learning for Particle Counting* (2025). The ML code base itself was
provided by the research group and is not mine to publish. These two files are the ones I wrote:

- `PipelineControl.py`: drives the whole pipeline (raw waveform conversion, automatic labelling, training
  data generation, training, evaluation) across window sizes.
- `Stats.py`: compares model predictions with manual labels and aggregates the error statistics reported
  in the thesis.

They import modules from that code base, so they do not run standalone.
