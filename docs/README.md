# SGL-JAX Documentation

The documentation tree uses two complementary formats:

- The main project docs use Sphinx from this `docs/` directory.
- The model cookbook under `docs/cookbook/` keeps a Mintlify-style `docs.json` and recipe layout.

Build the Sphinx docs locally:

```bash
python -m pip install -r docs/requirements.txt
make -C docs html
```

The Sphinx build intentionally excludes `docs/cookbook/` because the cookbook is organized for Mintlify navigation and model onboarding.
