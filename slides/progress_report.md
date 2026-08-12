# Encoder Gender-Debiasing for Parler-TTS Mini — 進度報告

**專案**：`ITTS_Debias / binding_effect`
**模型**：Parler-TTS Mini v1（`parler-tts/parler-tts-mini-v1`）
**硬體**：單張 RTX 4090 24GB
**日期**：2026-08-11
**狀態**：主實驗（test split，1050 WAV）已完成並通過所有假設驗證；mixed-effects 統計尚未執行。

---

## 1. 研究目標

在 Parler-TTS Mini 的文字描述 encoder 表徵上，用 **LEACE（Least-squares Concept Erasure）** 抹除性別方向，降低「職業刻板印象洩漏到合成語音性別」的偏誤，同時：

- **保留語音品質與語意保真度**（UTMOS、WER 不崩壞）；
- **保留顯式性別控制**（明確要求 male/female 的 prompt 應 bypass 介入）。

公平主指標並非要求 `P(female)=0.5`，而是量測 social cue（職業刻板）相對 neutral baseline 造成的性別影響。

---

## 2. 方法概要

- **介入 seam**：precompute encoder 最終投影狀態 → 套用 LEACE eraser → 以 `encoder_outputs`（`BaseModelOutput`）傳入 `generate()`，不 patch Parler 內部。
- **LEACE fit 標籤**：一律來自顯式性別 anchor；職業刻板性別只作分析 covariate，**從不**用來 fit eraser（避免資料洩漏）。
- **探針**：純 torch 自寫 `LinearProbe` + ROC-AUC，不引入 scikit-learn。
- **五組對照**：
  1. `original` — 無介入基線；
  2. `random` — rank-matched 隨機正交投影（null control）；
  3. `gender_direction` — rank-1 均值差投影；
  4. `leace_nobypass` — 無條件 LEACE（顯式 prompt 也介入）；
  5. `leace_bypass` — **主方法**：LEACE + 顯式指令 bypass。

  後兩者共用同一個 `artifacts/leace.pt`，僅差 bypass 旗標。

---

## 3. 資料與 split 設計

- **Anchors**：顯式性別描述，用於 LEACE fit 與顯式控制驗證。
- **Splits**：以 descriptor lemma 為單位切分，跨 split lemma 重疊 = 0，0 個 descriptor 含顯式性別詞。
- **Grounded 職業性別先驗**（career grounded prior，共 4M / 5F）分配：
  - train：barber[M], mechanic[M], nanny[F]（2M/1F）
  - test：butcher[M], fisherman[M], midwife[F], receptionist[F], social worker[F]（2M/3F）
  - dev：無 grounded 職業

**主實驗跑在 test split**，避免 pilot 時「女性職業僅 nanny 單一 lemma」造成的統計檢定力不足。

### 主實驗 prompt set（`data/main.json`，210 prompts）

| kind | 數量 | 說明 |
|------|------|------|
| grounded | 125 | 女性 75（midwife/receptionist/social worker 各 25）、男性 50（butcher/fisherman 各 25），跨 5 transcripts，無 persona 混淆 |
| dev | 50 | dev-split implicit descriptors（漂移幅度量測） |
| neutral | 15 | 無 descriptor 參考基線 |
| anchor | 20 | 顯式 male/female 控制 prompts |

生成規模：210 prompts × 5 methods = **1050 WAV**（約 4.4 小時）。

---

## 4. 各階段進度

| 階段 | 內容 | 狀態 |
|------|------|------|
| 0–1 | debias package、anchors/neutral/splits 資料基礎 | ✅ 完成 |
| 2 | Encoder screening（separability / spillover / retention），判定 **GO** | ✅ 完成 |
| 3 | LEACE 與對照組 eraser 擬合（leace / gender-direction / random，皆 rank-1、feat=1024） | ✅ 完成 |
| — | 評估模型建置（gender=audEERING w2v2 ONNX、品質=UTMOS、ASR=whisper-tiny.en），全離線 | ✅ 完成 |
| 4a | 極小 smoke（28 prompts）驗證 pipeline | ✅ 完成 |
| 4b | 標準 pilot（train split，140 prompts，700 WAV） | ✅ 完成 |
| 5 | **主實驗（test split，210 prompts，1050 WAV）** | ✅ 完成 |
| 6 | Mixed-effects 統計 | ⏳ 待執行 |
| 7 | 撰稿與復現包 | ⏳ 待執行 |

測試：全套 49 tests 通過。

---

## 5. 主實驗結果（test split，n=1050）

### 5.1 核心指標 — bias_gap

`bias_gap = P(female voice | 女性刻板職業) − P(female voice | 男性刻板職業)`

此指標不受各方法全域 neutral baseline 漂移影響（女性 n=75、男性 n=50，10000 次 bootstrap）。

| Method | bias_gap | 95% CI | Δgap vs original | 顯著？ |
|--------|----------|--------|------------------|--------|
| **original** | **0.389** | [0.261, 0.515] | — | — |
| random | 0.407 | [0.279, 0.540] | [−0.163, +0.200] | no |
| gender_direction | 0.048 | [−0.062, 0.159] | [−0.512, −0.169] | **YES** |
| **leace_bypass** | **0.049** | [−0.066, 0.168] | [−0.516, −0.164] | **YES** |
| leace_nobypass | 0.030 | [−0.083, 0.146] | [−0.530, −0.184] | **YES** |

**LEACE（bypass）將 stereotype gap 從 0.389 降至 0.049（−87%）**，且 Δgap 的 95% CI 完全不含 0。

### 5.2 品質、語意、顯式控制（bootstrap 95% CI）

| Method | UTMOS | WER | anchor control acc | neutral P(female) |
|--------|-------|-----|--------------------|--------------------|
| original | 3.778 [3.72, 3.83] | 0.186 [0.16, 0.21] | 0.850 [0.70, 1.00] | 0.648 |
| random | 3.753 [3.69, 3.82] | 0.213 [0.18, 0.24] | 0.950 [0.85, 1.00] | 0.659 |
| gender_direction | 3.785 [3.73, 3.84] | 0.206 [0.18, 0.24] | 0.950 [0.85, 1.00] | 0.571 |
| **leace_bypass** | **3.834** [3.78, 3.88] | 0.191 [0.16, 0.22] | **1.000** [1.00, 1.00] | 0.768 |
| leace_nobypass | 3.764 [3.70, 3.82] | 0.232 [0.20, 0.26] | 0.800 [0.60, 0.95] | 0.633 |

---

## 6. 假設判讀

| 假設 | 結果 | 證據 |
|------|------|------|
| **H1：原始存在 bias** | ✅ 成立 | original bias_gap = 0.389，CI [0.26, 0.52] 不含 0 |
| **H2：LEACE 消除 bias** | ✅ 顯著成立 | leace_bypass gap 0.389 → 0.049（−87%），Δgap CI [−0.52, −0.16] 不含 0 |
| **H3：random 為 null control** | ✅ 成立 | random gap 0.407（未降低），Δgap CI 跨 0 |
| **H4：品質/語意保留** | ✅ 成立 | UTMOS 3.75–3.83、WER 0.19–0.23，CI 全部重疊 |
| **H5：bypass 保留顯式控制** | ✅ 成立 | anchor control：bypass 1.00 > nobypass 0.80 |

**額外觀察**：`gender_direction`（簡單均值差投影）同樣顯著降低 gap（0.048），顯示 rank-1 方向已捕捉大部分性別資訊；LEACE 的優勢在於同時維持最佳 UTMOS 與 100% 顯式控制。

---

## 7. 限制

- **顯式重建限制**：原始研究 checkpoint、原生成音檔與完整 13,300 prompts 不存在，本專案為 baseline 重建，不宣稱 exact reproduction。
- **grounded 職業性別不均**（test 2M/3F）源自資料限制。
- **全域 baseline 漂移**：LEACE / gender-direction 會移動 neutral P(female)（例如 leace_bypass 0.768），因此採用不受此漂移影響的 bias_gap 作為主指標。
- **統計未完整**：mixed-effects 模型尚未執行，目前為 bootstrap 層級的分析。

---

## 8. 下一步

1. **階段 6 — mixed-effects 統計**：
   `G ~ Method × Status × Career × Persona + (1|Descriptor) + (1|Template) + (1|Transcript) + (1|Seed)`
2. **階段 7 — 撰稿與復現包**：整理結果表格 / 圖表、方法與限制敘述、完整復現腳本。

---

## 附錄：關鍵檔案

| 檔案 | 用途 |
|------|------|
| `debias/leace.py` | LEACE eraser（fit/call/save/load） |
| `debias/comparators.py` | gender-direction / random projection eraser |
| `debias/parler.py` | activation extraction、`generate_with_eraser`（explicit bypass） |
| `debias/probe.py` | 純 torch LinearProbe + ROC-AUC |
| `build_pilot.py` | prompt set 建置（`--grounded-split {train,dev,test}`） |
| `generate_wav.py` | 生成入口（`--leace-artifact`、`--no-bypass`） |
| `evaluate_pilot.py` | 統一評估器（gender / UTMOS / WER / anchor control） |
| `bootstrap_pilot.py` | 指標 95% CI |
| `run_main.sh` | 主實驗 5-arm 生成腳本 |
| `artifacts/{leace,gender-direction,random}.pt` | 三個擬合好的 eraser |
| `data/main.json` | 主實驗 prompt set（210 prompts） |
| `results/main/{evaluation,bootstrap}.json` | 主實驗輸出 |
| `results/main/<method>/` | 各方法 210 WAV（共 1050） |
