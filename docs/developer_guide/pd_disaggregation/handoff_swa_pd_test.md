# SWA PD 分離 実装 & テスト Handoff

**日期**: 2026-07-04
**分支**: `epic/mimo-pd-disggragation` (已 push)
**作者**: jiongxuan + Claude

---

## 1. コード変更概要

### 目標
sglang-jax の PD 分離モードで、SWA (Sliding Window Attention) hybrid モデル（MiMo-V2-Flash）の KV transfer をサポートする。

### 核心設計
- raiden (tpu-raiden) の `register_read` は全 layer に同じ block_ids をブロードキャストする。SWA モデルでは full layer と SWA layer で異なる page index space を使うため、**2 つの KVCacheManager**（full 用 + SWA 用）を作成
- SWA layer は sliding window 尾部のみ転送（全 token は転送しない）、`full_to_swa_index_mapping` で index 翻訳

### 変更ファイル (7 files, 6 modified + 1 test)

| ファイル | 変更内容 |
|----------|---------|
| `srt/disaggregation/jax_transfer/wrapper.py` | `RaidenTransferWrapper` に `_engine_full` + `_engine_swa` のデュアル KVCacheManager サポート。`register_read`/`start_read`/`poll_stats` を両エンジンに透過 dispatch |
| `srt/disaggregation/prefill.py` | 新規 `_extract_swa_block_ids_for_chunk` — `full_to_swa_index_mapping` で index 翻訳し、sliding window 尾部のみ抽出。`_raiden_handoff_chunk` で SWA blocks も登録 |
| `srt/disaggregation/decode.py` | `_admit_one_raiden` で `swa_local_pages` + `swa_remote_endpoint` を構築し PMetadata 経由で receiver に渡す |
| `srt/disaggregation/jax_transfer/conn.py` | `PMetadata` に `swa_remote_endpoint`/`swa_local_pages` 追加。`send_chunk`/`producer_register_read`/`_poll_raiden` で SWA パラメータを透傳 |
| `srt/disaggregation/bootstrap.py` | `RegisterTransferRequest`/`BootstrapClient.register_transfer` に `swa_block_ids`/`swa_raiden_endpoints_json` 追加 |
| `srt/disaggregation/runtime.py` | `SWAKVPool`（`hasattr(pool, "full_kv_pool")`）の検出時に、`kv_caches_swa` を raiden wrapper に渡す |
| `test/test_pd_swa_basic.py` | CPU 側単体テスト 8 件。tail filtering / chunk boundary / 非 SWA 後方互換 |

### コミット
```
c68700b2b test(pd/swa): CPU 侧单元测试验证 SWA block extraction 逻辑
34920af14 feat(pd/swa): raiden 双引擎支持 SWA hybrid attention 模型的 KV transfer
```

---

## 2. 検証状態

| 項目 | 環境 | 結果 |
|------|------|------|
| CPU 単体テスト | Mac | ✅ 8/8 通過 |
| 非 SWA 回帰テスト (DeepSeek-1.5B) | GKE v6e-1, raiden PD | ✅ GSM8K 0.67, TTFT 正常, OOM 0, `is_hybrid_swa=False` |
| raiden v7x キャッシュ構築 | Falcon v7x-8 | ✅ ビルド + `gs://inference-model-storage-poc-tpu-hns/raiden-cache/raiden-v7x-jax0.10.2.tar.gz` に保存 |
| MiMo-V2-Flash モデルダウンロード | Falcon v7x-8 | 🔄 60/157 ファイル完了、残りダウンロード中 |
| SWA PD prefill-only テスト | Falcon v7x-8 | ⏳ モデルダウンロード完了後に実施 |

---

## 3. サーバー/クラスタ状態

### Falcon クラスタ (tpuv7x-64-node)
- **クラスタ ID**: `cl-ivvc22wike`
- **プロジェクト**: `poc-tpu-partner`
- **GCS バケット**: `gs://inference-model-storage-poc-tpu-hns`
- **v7x-8 設定**: `device_type=v7x, device_count=8, device_topo=2x2x1, replica=1`
- **利用可能イメージ**: `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1` (JAX 0.10.2 にアップグレード必要)

### GCS バケット構成
```
gs://inference-model-storage-poc-tpu-hns/
├── raiden-cache/
│   └── raiden-v7x-jax0.10.2.tar.gz    # 27MB, raiden プリビルド
├── MiMo-V2-Flash/                      # ダウンロード中 (60 files)
│   ├── config.json
│   ├── model-*-of-*.safetensors
│   └── ...
└── experiments/
    └── exp-*/
```

### 実行中/履歴 Falcon 実験

| exp_id | 状態 | 説明 |
|--------|------|------|
| `exp-yllelfoc4z` | running | MiMo ダウンロード再開中 (HF→GCS) |
| `exp-dgvv0u0olj` | running | Qwen3-8B で SIGABRT（SWA 非該当） |
| `exp-jfuzxv6rr1` | failed | raiden 構築成功したがモデル DL 失敗（pod 死亡） |
| `exp-3hyx4untlm` | pending | GCS 権限エラーで stuck |
| その他 | failed | pipefail / JAX version 問題 |

---

## 4. 操作方法

### Falcon ジョブの状態確認
```bash
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/opt/homebrew/share/google-cloud-sdk/bin:/usr/bin:$PATH"

# 特定の実験の状態
falcon exp get <exp_id> --output json | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(d['status'], d['job_status'])
for c in d.get('job_conditions',[])[-3:]:
    print(f'  [{c[\"type\"]}] {c[\"status\"]} - {c.get(\"message\",\"\")[:150]}')
"

# ログ確認
falcon exp logs <exp_id> --tail 30

# Pod でコマンド実行
falcon exp exec <exp_id> -- <command>

# 実験一覧
falcon exp list --limit 10 --cluster tpuv7x-64-node --output json
```

### マニフェスト提出
```bash
# マニフェスト作成 → /tmp/falcon-swa-v2.yaml に保存
# 提出
falcon workflow profile submit -f /tmp/falcon-swa-v2.yaml --output json
# 待機
falcon workflow profile collect <exp_id> --timeout 60m --output json
```

### GCS 操作 (pod 内)
```bash
# raiden キャッシュ確認
gsutil ls gs://inference-model-storage-poc-tpu-hns/raiden-cache/

# MiMo モデル確認
gsutil ls gs://inference-model-storage-poc-tpu-hns/MiMo-V2-Flash/ | wc -l

# HF → GCS モデルダウンロード再開
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('XiaomiMiMo/MiMo-V2-Flash',
    local_dir='/tmp/mimo-model',
    local_dir_use_symlinks=False,
    resume_download=True, max_workers=8)
"
gsutil -m rsync -r /tmp/mimo-model gs://inference-model-storage-poc-tpu-hns/MiMo-V2-Flash/
```

---

## 5. SWA PD テスト計画

### テスト 1: Prefill-Only (1 台 v7x-8)
**目標**: SWA コードパスの基本動作確認

```yaml
# マニフェスト key points:
# - model-path: /models/MiMo-V2-Flash (GCS mount)
# - tp-size: 8, dp-size: 1 (PD 非対応のため DP=1)
# - ep-size: 8, moe-backend: fused_v2
# - disaggregation-mode: prefill
# - disaggregation-use-raiden: true
# - swa-full-tokens-ratio: 0.2
# - マウント:
#     gs://.../MiMo-V2-Flash → /models/MiMo-V2-Flash (read-only)
#     gs://.../raiden-cache → /models/raiden (read-only)
```

**検証項目**:
1. `is_hybrid_swa=True, swa_layer_num=39` (prefill server log)
2. `RaidenTransferWrapper started ... is_hybrid_swa=True` 
3. `RAIDEN-P register_read ... n_swa>0`
4. 0 OOM / 0 Traceback

### テスト 2: Full PD E2E (2 台 v7x-8)
**目標**: prefill + decode の完全な SWA KV transfer

2 台必要。DP attention の PD 対応が完了してから実施予定。

---

## 6. 既知の問題と注意点

1. **DP attention は PD 未対応**: `dp_size > 1` は動作しない。`dp/support_dp` ブランチで準備中だが E2E 未検証
2. **cross-project GCS 権限**: `model-storage-sglang` (tpu-service-473302) は falcon (poc-tpu-partner) からアクセス不可。falcon のバケット `inference-model-storage-poc-tpu-hns` を使うこと
3. **イメージの JAX バージョン**: jax0.9.0-rev1 イメージは JAX 0.9.0。0.10.2 にアップグレードするには `pip install 'jax[tpu]==0.10.2'` が必要
4. **MiMo-V2-Flash のサイズ**: 総重量 291GB (256 experts × 48 層の MoE)。ダウンロードに数時間。`resume_download=True` と `gsutil -m rsync` で断点再開可能
5. **raiden キャッシュ**: v6e と v7x で異なるアーキテクチャのため別途ビルドが必要。v7x 用キャッシュは `gs://inference-model-storage-poc-tpu-hns/raiden-cache/` にあり
