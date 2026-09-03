# Asset manifests

These YAML files are the lightweight, reviewable catalog for simulation
assets. They contain IDs, provenance, revisions, paths, task semantics and
licensing status; they do not contain meshes, textures, checkpoints or scene
XML copied from another project.

## Rules

1. Every entry gets a stable dotted `id` and an explicit `version`.
2. `source.uri`, `source.revision` and the original relative `source.path` are
   required for anything not created in this repository.
3. `license.spdx` must be an SPDX identifier once verified. Until then use
   `UNKNOWN` and set `license.redistribution` to a restrictive status such as
   `prohibited_until_upstream_license_review`.
4. `paths` are repository-relative and point to downloaded/ignored files. A
   clean checkout must still pass metadata validation; use `--check-files`
   only after downloading the relevant asset group.
5. Do not commit a third-party mesh/XML merely because it is referenced by a
   manifest. Preserve the upstream license and notices when redistribution is
   explicitly permitted.

For a downloaded file, add `integrity.sha256` after the upstream revision is
fixed. This makes local downloads auditable without putting the binary into
Git. A future asset release should publish the same digest in its download
manifest.

Validate the catalog with:

```bash
python scripts/dev/validate_asset_manifests.py
python scripts/dev/validate_asset_manifests.py --check-files
```

Install the robosuite v1.5.2 tabletop objects (bottle, can and lemon) with:

```bash
.venv/bin/python scripts/setup/download_scene_object.py
```

The command verifies every download against a pinned SHA256 and writes the
ignored `assets/objects/robosuite/<object>/v1/` directory plus a small MuJoCo
body fragment. Pass `bottle`, `can` or `lemon` to install one object. The
upstream MIT `LICENSE` is downloaded alongside each object.

Build a complete 43-DOF scene containing an imported object:

```bash
.venv_scene/bin/python scripts/setup/build_scene_with_object.py <object>
```

The generated XML is ignored under `outputs/scenes/`; launch it with the
existing scene runtime's `--scene-xml` option (or use the aliases
`--scene robosuite-bottle`, `--scene robosuite-can`, and
`--scene robosuite-lemon`).

Validate the generated fragment itself with:

```bash
.venv_scene/bin/python scripts/dev/validate_scene_object.py <object>
```

The first command is suitable for CI on a clean checkout. The second also
checks that downloaded scene/robot paths exist locally.
