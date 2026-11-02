# Tests

The test suite runs with [pytest](https://docs.pytest.org) inside the `venv_columnar_dev` sandbox (which ships pytest). Run it from a shell after `source setup.sh <setup_name>`:

```shell
./tests/run_tests                          # everything except the NAF tier
./tests/run_tests tests/unit               # a directory, file or single test, e.g. tests/unit/test_util.py::HbwUtilTest
./tests/run_tests -m "not config"          # only the fast tests that need no analysis config
./tests/run_tests -k lepton -x             # any further pytest options
bash tests/run_test columnar/test_selection.py
./tests/run_naf                            # NAF tier (cvmfs, dust, law CLI)
./tests/run_all                            # linting + tests (as in CI)
```

`run_tests` uses the tracked `law.cfg` (override with `HBW_TEST_LAW_CONFIG`), so results do not depend on a personal law config.

## Layout

| Directory | What is tested | Needs |
|-----------|----------------|-------|
| `unit/` | pure helpers (`hbw/util.py`, `hist_util.py`, inference and trigger helpers, process helpers) | nothing |
| `columnar/` | categorizers, selectors and producers on small synthetic events with known results | toy config; `config` marker for tests on the real config |
| `config/` | `law*.cfg` module lists, config building, defaults, ML and inference model registries | `config` marker (builds configs) |
| `tasks/` | law tasks are instantiated with the analysis defaults and their requirements resolved, without running anything | `config` marker |
| `naf/` | external files exist, `law run ... --print-deps` works | `naf` marker, only on the NAF |

Shared helpers live in `helpers.py` (`toy_config`, `make_events`, `run_taf`, ...) and `conftest.py` (the session-scoped `get_config(analysis, config)` fixture).

## Markers

- `config`: the test builds analysis configs. The first config of a campaign requires its pickled campaign from `hbw.BuildCampaignSummary`, which the `get_config` fixture creates when missing (only needs cmsdb, a few minutes). Set `HBW_TEST_STORE=<dir>` to write it into a separate store instead of `$CF_STORE_LOCAL`.
- `naf`: needs DESY NAF resources; skipped by default (`pytest.ini`), run with `-m naf` or `./tests/run_naf`.

## Known issues

Problems that are known but not yet fixed are listed in `known_issues.py`. Their tests are strict xfails: they do not fail the suite while the problem persists, but they do fail (XPASS) once it is fixed, so that the entry can be removed.

## Writing tests

- Columnar modules: build events with `make_events({...}, behavior=True)` and call the module with `run_taf(module, events, config_inst, dataset, **call_kwargs)`. Selectors return `(events, results)`, categorizers `(events, mask)`. Use `toy_config(**aux)` when the module only needs a few aux entries, otherwise the real config via `get_config`.
- Choose a handful of events per cut, including edge cases (values at thresholds, empty collections), and assert exact results.
