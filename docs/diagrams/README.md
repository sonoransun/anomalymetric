# Mermaid diagrams

Sources for every Mermaid diagram embedded in the documentation live here. Each
`*.mmd` file is the source of truth and is **also** inlined as a fenced
```mermaid``` block in the relevant markdown page so that GitHub renders the
diagram without any build step.

| File | Used in |
| --- | --- |
| `module-deps.mmd` | `CLAUDE.md`, `docs/for-contributors/architecture.md` |
| `pipeline-flow.mmd` | `README.md`, `docs/for-users/quickstart.md` |
| `class-diagram.mmd` | `docs/for-contributors/architecture.md` |
| `spectrumkind-dispatch.mmd` | `docs/for-users/concepts.md`, `docs/for-contributors/design-decisions.md` |
| `loeb-turner-sequence.mmd` | `docs/for-users/scoring.md` |

## Editing

Edit the `.mmd` source, then update the inlined block in each consumer page so
GitHub's renderer stays in sync. Validate by pasting the source into
[mermaid.live](https://mermaid.live) or running `mmdc -i <file>.mmd -o /tmp/preview.svg`
if `mermaid-cli` is installed.

When the mkdocs site is built, `mkdocs-mermaid2-plugin` renders the inlined
fences directly — no `.mmd` files are read at build time.
