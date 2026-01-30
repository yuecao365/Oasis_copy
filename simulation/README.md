# simulation – Oasis Twitter simulation

Run the Oasis Twitter simulation from here. Use the config file to point at your own agent data.

## Submodule

The **`oasis`** package comes from the [camel-ai/oasis](https://github.com/camel-ai/oasis) submodule at `oasis/`. The script adds the submodule root to `sys.path` so `import oasis` uses `oasis/oasis/`.

- Init/update: `git submodule update --init --recursive`
- Install deps from the submodule (run from repo root):
  - `pip install -e oasis` **or** `cd oasis && poetry install`
- Default data (when submodule exists) is under `oasis/data/`. Config paths are relative to repo root.

## Run

From the **repo root**:

```bash
python simulation/twitter_simulation.py --config_path simulation/config.yaml
```

With default config (same file):

```bash
python simulation/twitter_simulation.py
```

## Config

Edit **`simulation/config.yaml`** to use your own data:

- **`data.csv_path`** – Agent CSV. Required columns: `user_id`, `name`, `username`, `following_agentid_list`, `previous_tweets`, `user_char`, `description`. Path is relative to repo root.
- **`data.db_path`** – SQLite DB path for this run (created fresh each run). Use `:memory:` for in-memory. Path relative to repo root.
- **`simulation.num_timesteps`**, **`clock_factor`**, **`recsys_type`**, **`available_actions`** – Simulation settings.
- **`inference.model_type`** – e.g. `gpt-4o-mini`. Ensure `OPENAI_API_KEY` is set.

## Layout

- `twitter_simulation.py` – Simulation entrypoint.
- `config.yaml` – Config; change `csv_path` and `db_path` to your data.

Logs go to `social.log` in the current directory (run from repo root).
