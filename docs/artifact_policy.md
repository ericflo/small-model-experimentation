# Artifact Policy

## In Git

Keep these in git when they are reasonably sized:

- README files and reports,
- source and scripts,
- configs and manifests,
- small CSV/JSON/JSONL result tables,
- plots and report figures,
- small reproducible datasets.

## In Git LFS

Use Git LFS for model-like binaries and future large artifacts:

- `*.safetensors`
- `*.pt`
- `*.pth`
- `*.ckpt`
- `*.bin`
- `*.onnx`
- `*.arrow`
- `*.parquet`

Do not check trained adapter output directories into the repository. Keep paths such as `experiments/**/reports/adapters/` local or document them in an external artifact manifest.

## External Storage

For artifacts too large or awkward for the repository, add a manifest inside the experiment. New experiments should use `reports/artifact_manifest.yaml`; imported experiments may also have historical `large_artifacts_manifest.md` or `checkpoint_manifest.csv` files.

Include:

- artifact name,
- external path or storage location,
- checksum if available,
- command or process to regenerate it,
- whether it is required for reproduction or only for inspection.

## Standard Manifest

New experiment scaffolds include:

```text
reports/artifact_manifest.yaml
```

Use it for:

- trained adapters kept outside git,
- large checkpoints or external cache paths,
- omitted local output directories,
- smoke and full reproduction commands,
- required external artifacts for reruns.

The generated [artifact manifest index](../knowledge/artifact_manifest_index.md) lists repository manifests across experiments.
