# 安装、运行和复现

## 环境

验证设备：Jetson AGX Orin 64GB，aarch64 / Python 3.12.3 / vLLM 0.9.2 V1 / Qwen3-4B-Instruct。模型路径通过 `MODEL` 设置，服务名固定 `qwen3-4b`。本仓库不提供模型权重或替代 Jetson 的 vLLM 安装流程。

在实际启动 vLLM 的 Python 环境中执行：

```bash
python3 -c 'import sys, vllm; print(sys.executable, vllm.__version__)'
python3 -m pip install -r requirements-bench.txt
python3 manage_patch.py check
python3 scripts/run_tests.py
# 确认旧服务已停止
python3 manage_patch.py install
python3 tests/test_retention.py --installed
```

安装器要求五个源文件的 SHA256 与 `manifest.json` 一致。不匹配时不要手动覆盖或修改 manifest 绕过检查；需要先适配实际源码。备份位于 vLLM 包内 `.prefix_retention_backup_v1/`。

若实际解释器为 `/opt/venv/bin/python3`，激活对应环境，或在脚本命令前设置 `PYTHON_BIN=/opt/venv/bin/python3`。无需为补丁重装 CUDA/PyTorch。

## 启动与自检

```bash
# 终端 A
bash start_server.sh lru 2048
# 或在停止 LRU 后：bash start_server.sh slru 2048
```

固定参数：BF16、eager、max-model-len=16384、max-num-seqs=8、max-num-batched-tokens=16384、block-size=16、gpu-memory-utilization=0.60。`--num-gpu-blocks-override=2048` 控制实际容量。Prefix caching 与 prompt token 明细开启。

脚本将服务绑定到 `127.0.0.1:8000`，开启实验需要的 `VLLM_SERVER_DEV_MODE=1`。它是隔离 benchmark 服务，reset 操作会清空缓存；不要对承载其他请求的服务执行 benchmark。容器端口访问时按需设置 `HOST=0.0.0.0` 和 Docker 端口映射。

以服务日志确认实际配置：

```text
PREFIX_RETENTION policy=lru
# 或
PREFIX_RETENTION policy=slru protected_ratio=0.5 protected_limit=1023
GPU KV cache size: 32,768 tokens
```

```bash
# 终端 B，等待服务就绪
curl -f http://127.0.0.1:8000/health
python3 smoke_server.py --label lru --strict-output
RUN_ID=pilot_r1 bash run_bench.sh lru 2048 42
```

SLRU 服务下将客户端标签改为 slru。自检默认生成一个 token，检查 cold → hit → reset-cold，并分别报告缓存行为和输出一致性。单 token 匹配不等于证明所有长输出完全一致。客户端兼容 `/server_info` 404；reset 接口和 cached token 明细必须可用。

## 长测复现

LRU 服务下：

```bash
for SEED in 42 43 44; do
  WARMUP=200 REQUESTS=1000 CONCURRENCY=1 RUN_ID=formal_r1 \
    bash run_bench.sh lru 2048 "$SEED" || break
done
```

停止 LRU，启动 SLRU 后：

```bash
for SEED in 42 43 44; do
  WARMUP=200 REQUESTS=1000 CONCURRENCY=1 RUN_ID=formal_r1 \
    bash run_bench.sh slru 2048 "$SEED" || break
done
for SEED in 42 43 44; do
  python3 compare_results.py \
    "results/lru_b2048_s${SEED}_formal_r1" \
    "results/slru_b2048_s${SEED}_formal_r1" || break
done
```

每个结果目录只能创建一次；失败后改用 `RUN_ID=formal_r2`，保留原目录。`run_bench.sh` 不负责启动或切换服务。对比工具检查 prefix pool、请求 trace 哈希和客户端配置；真实策略与容量仍需核对引擎日志。

## 可选硬件与 ABA 诊断

宿主机采集器只查询功耗与时钟设置，不修改频率或风扇。它会记录进程命令行，因此原始遥测上传前应自行检查。

```bash
# Jetson 宿主机；将此路径换成宿主机实际仓库位置
sudo python3 /path/to/vllm-prefix-retention/diagnostics/collect_jetson.py \
  --out /tmp/jetson_retention_r1
```

停止手动启动的服务后，在 vLLM 所在环境、仓库根目录运行：

```bash
python3 diagnostics/run_aba.py --root "$PWD" --out /tmp/retention_aba_r1
```

ABA 工具自行启动/停止 LRU → SLRU → LRU；不会接管已占用端口。默认每阶段 5 次冷预热、30 次冷测量，随后 50 次 Zipf 预热、100 次测量。完成后用 Ctrl+C 停止宿主机采集器。

Docker 宿主机 `/tmp`、宿主机 `~/tmp` 和容器 `/tmp` 是不同位置。若需要复制结果，在宿主机执行：

```bash
docker cp vllm:/tmp/retention_aba_r1 ./retention_aba_r1
```

## 回滚

```bash
# 先停止该 vLLM 服务
python3 manage_patch.py restore
python3 manage_patch.py check
```

回滚还原原始 BlockPool 并删除新增模块，保留备份。若安装后文件被其他工具改过，恢复会拒绝覆盖。恢复后使用原来的 `vllm serve` 命令；本仓库启动脚本要求补丁存在。只切回基线时，无需卸载补丁，重启 `bash start_server.sh lru 2048` 即可。
