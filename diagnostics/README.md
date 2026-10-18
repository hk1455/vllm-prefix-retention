# Runtime diagnostics

- `collect_jetson.py`: read-only Jetson host telemetry via tegrastats, nvpmodel, jetson_clocks and process snapshots. Run on the host even if vLLM runs in Docker.
- `run_aba.py`: start separate LRU / SLRU / LRU server processes; verify engine policy/capacity; run fixed cold calibration and Zipf requests. Pass `--root` with the actual repository directory.

See [运行指南](../docs/RUNNING.md) for commands. These tools do not install the patch or change device clock settings. Raw process snapshots can include local command-line information; review them before sharing.
