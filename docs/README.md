# SGL-JAX Documentation

The documentation tree uses two complementary formats:

- The main project docs use Sphinx from this `docs/` directory.
- The model cookbook under `docs/cookbook/` keeps a Mintlify-style `docs.json` and recipe layout.

## Main Docs (Sphinx)

Build and preview the Sphinx docs locally from the repository root:

```bash
python -m pip install -r docs/requirements.txt
make -C docs html
python -m http.server 8000 --directory docs/_build/html
```

Open <http://localhost:8000>.

The Sphinx build intentionally excludes `docs/cookbook/` because the cookbook is organized for Mintlify navigation and model onboarding.

## Cookbook (Mintlify)

Preview the cookbook locally with Mintlify in a separate terminal:

Prerequisites:

- Node.js >= 20
- npm

Install the Mintlify CLI and start the cookbook dev server:

```bash
npm i -g mint
cd docs/cookbook
mint dev --port 3000
```

Open <http://localhost:3000>.

Use a Node.js LTS release if the CLI rejects newer non-LTS Node.js versions.

## Ports

The Sphinx preview uses port 8000 and the cookbook preview uses port 3000, so they do not conflict when you run both commands above. If either port is already in use, choose a different explicit port, for example `python -m http.server 8001 --directory docs/_build/html` for Sphinx or `mint dev --port 3333` for the cookbook.
