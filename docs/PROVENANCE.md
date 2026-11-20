# Provenance

## Bootstrap state

This repository bootstrap contains packaging and developer tooling only. It has not copied
source code, datasets, generated artifacts, or ontology assets from another repository.

## Planned dataset source

The separately authorized dataset migration is expected to use:

- Repository: <https://github.com/midhunpradeep/town_model>
- Branch: `main`
- Pinned source commit: `1788acfc65eb1038fe9e4c595cab6ef0f25407fd`

When migration occurs, update this document with a manifest of migrated paths, the pinned
source revision, material changes made for packaging/imports, and the validation performed.
Record each third-party asset's own origin and license as well; repository provenance does
not replace asset-level attribution.

The pinned source checkout has no top-level project license file. Its
`docs/licenses/UPSTREAM_PIZZAIOLO.md` records that the pinned upstream Pizzaïolo project
revision `6c0bcb221a957c184012beaf103bebcde4b1f793` declares CC BY-NC 4.0. Review that notice
and all relevant asset terms before migrating or redistributing any such material. Do not
assume that the upstream asset license covers the source repository's own code.

## Project license status

No project-wide license has been selected for this new repository, and this bootstrap does
not grant or infer licensing rights for future migrated files. Resolve project and source
file licensing with the rights holder before publishing or redistributing copied material.
See [`licenses/README.md`](licenses/README.md) for the recording checklist.
