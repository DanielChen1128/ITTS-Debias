# Parler Mini / Large Constant Steering 2x：完整實驗報告

日期：2026-08-31

## Slide 1｜Parler Mini / Large Constant Steering 2x

- 相同方法、相同strength、分別fit model-local steering direction。
- 每個模型完成13,300個bias prompts與500組paired quality samples。
- 報告global calibration、axis-level分布、binding interaction與自動品質指標。

## Slide 2｜本次實驗回答的問題

1. 相同的female-to-male constant steering 2x能否跨Mini與Large降低global female skew？
2. 介入後，career、persona、status與compositional prompts是否呈現一致變化？
3. 64個matched cells的binding interaction magnitude如何改變？
4. UTMOS與Whisper WER在500組paired samples上的變化幅度為何？

本次重點是完整report觀察值，不將單一自動品質判定等同於主觀可感知品質。

## Slide 3｜共同實驗設定

| Item | Parler Mini | Parler Large |
|---|---|---|
| Model revision | `0392b945...810d9` | `50cb4b87...4f11e` |
| Encoder width | 1,024 | 1,536 |
| Method | Female-to-male constant centroid steering | Same |
| Strength | 2x | 2x |
| Intervention | Pooled final encoder representation | Same |
| Explicit gender prompts | Regex bypass | Regex bypass |
| Generation | Batch 8, fixed prompts and seeds | Same |

- Steering tensor依模型分別fit，不跨模型共用。
- Gender outcome固定比較`female_score`與`male_score`；tie判為female。

## Slide 4｜完成規模與分析流程

| Component | Mini | Large |
|---|---:|---:|
| Single-axis prompts | 6,900 | 6,900 |
| Compositional prompts | 6,400 | 6,400 |
| Bias evaluation total | 13,300 | 13,300 |
| Matched quality pairs | 500 | 500 |
| Interaction cells | 64 | 64 |
| Randomizations/cell | 10,000 | 10,000 |

流程：fixed prompts → Original/2x WAV → binary acoustic gender → interaction analysis → paired UTMOS/WER。

## Slide 5｜Global female rate

| Model | Original | Steering 2x | Change | Distance to 50% |
|---|---:|---:|---:|---:|
| Mini | 78.23% | 56.26% | −21.97 pp | 28.23 → 6.26 pp |
| Large | 74.73% | 51.37% | −23.36 pp | 24.73 → 1.37 pp |

- 兩個模型的global female rate都往50%移動。
- Large的global結果更接近50%；Mini仍保留約6.3 pp female skew。
- Global rate是所有prompt sets的aggregate，不代表每個axis都同時接近50%。

## Slide 6｜Axis-level female rate

| Prompt set | Mini Original | Mini 2x | Large Original | Large 2x |
|---|---:|---:|---:|---:|
| Status | 80.50% | 52.50% | 58.00% | 30.00% |
| Career | 79.56% | 60.70% | 88.37% | 72.48% |
| Persona | 77.05% | 49.68% | 85.58% | 61.30% |
| Two-axis | 78.94% | 59.59% | 64.05% | 40.23% |
| Three-axis | 77.72% | 57.62% | 61.41% | 33.62% |

- Mini在五個prompt sets都降低female rate，persona最接近50%。
- Large的career仍偏female；status與compositional prompts移向male側。
- 同一個2x strength在不同模型與axis上的位移量不同。

## Slide 7｜Binding interaction magnitude

| Model | Original mean `|I|` | Steering 2x mean `|I|` | Reduction | Strong interactions |
|---|---:|---:|---:|---:|
| Mini | 1.7151 | 0.7431 | 56.7% | 11 → 0 |
| Large | 3.2581 | 1.3546 | 58.4% | 37 → 3 |

| Model | Moderate before | Moderate after | Not significant after |
|---|---:|---:|---:|
| Mini | 19 | 10 | 54/64 |
| Large | 13 | 20 | 41/64 |

- 兩模型的平均interaction magnitude均降低超過一半。
- 結果支持interaction attenuation；仍保留部分moderate或strong cells。

## Slide 8｜Mini interaction family

| Family | Original mean `|I|` | Mini 2x | Change |
|---|---:|---:|---:|
| Career + persona | 1.5072 | 0.8866 | −41.2% |
| Status + career | 1.9738 | 0.9528 | −51.7% |
| Status + career + persona | 1.7386 | 0.6860 | −60.5% |
| Status + persona | 1.7777 | 0.4744 | −73.3% |

- 四個interaction families都呈現較低mean absolute magnitude。
- Mini 2x之後沒有strong interaction，仍有10個moderate cells。

## Slide 9｜500-pair品質評估

- 每個模型固定500組Original/2x paired samples。
- 五個strata各100組：status、career、persona、two-axis、three-axis。
- 指標：UTMOS作為自動語音品質proxy；Whisper WER作為內容可辨識度proxy。
- 報告paired mean delta與95% bootstrap CI。
- 自動指標不直接等同於人工聽感，以下以數值變化描述，不使用品質好壞的二元結論。

## Slide 10｜Mini品質指標變化

| Metric | Original | Mini 2x | Paired delta [95% CI] |
|---|---:|---:|---:|
| UTMOS | 3.8507 | 3.8186 | −0.0321 [−0.0777, +0.0142] |
| WER | 0.1752 | 0.1750 | −0.0002 [−0.0185, +0.0184] |

| Stratum | UTMOS delta | WER delta |
|---|---:|---:|
| Status | −0.0935 | +0.0137 |
| Career | +0.0278 | −0.0214 |
| Persona | −0.0034 | −0.0094 |
| Two-axis | −0.0985 | +0.0183 |
| Three-axis | +0.0070 | −0.0021 |

Mini的global WER幾乎不變；UTMOS差異依stratum正負皆有。

## Slide 11｜Large品質指標變化

| Metric | Original | Large 2x | Paired delta [95% CI] |
|---|---:|---:|---:|
| UTMOS | 3.5083 | 3.3919 | −0.1163 [−0.1916, −0.0397] |
| WER | 0.2931 | 0.3516 | +0.0586 [+0.0026, +0.1452] |

| Stratum | UTMOS delta | WER delta |
|---|---:|---:|
| Status | −0.1668 | +0.1843 |
| Career | −0.1203 | +0.0275 |
| Persona | −0.1689 | +0.0470 |
| Two-axis | −0.1293 | +0.0381 |
| Three-axis | +0.0036 | −0.0040 |

Large的平均差異較Mini明顯，且主要集中在部分strata；three-axis兩項指標接近不變。

## Slide 12｜品質數值的解讀範圍

- UTMOS與單一Whisper WER是automated proxies，可能受說話速度、音高、停頓與ASR錯誤型態影響。
- 500-pair paired design提高平均差異的估計精度，但不能回答差異是否可由聽者穩定察覺。
- Aggregate mean會混合不同strata；Large結果顯示status、persona與three-axis的變化模式不同。
- 後續可加入blind paired listening、較大型ASR及第二個speech-quality metric，區分metric shift與perceptual change。

## Slide 13｜跨模型觀察

| Observation | Mini 2x | Large 2x |
|---|---|---|
| Global female rate | 78.23% → 56.26% | 74.73% → 51.37% |
| Axis一致性 | 五組都降低，persona接近50% | 不同axis位移差異較大 |
| Mean `|I|` | −56.7% | −58.4% |
| Automated quality | Global差異小 | 平均差異較Mini明顯、具stratum差異 |

- 相同演算法與strength在兩個模型都降低global skew與interaction magnitude。
- 模型scale會改變steering response；global calibration、axis balance與品質指標需分開報告。

## Slide 14｜目前結論與後續工作

### 目前觀察

- Mini與Large的global female skew都降低。
- 64-cell mean interaction magnitude分別降低56.7%與58.4%。
- Mini 2x的500-pair UTMOS/WER平均變化有限。
- Large的自動品質差異集中於特定strata，尚需搭配聽測解讀實際影響。

### 後續工作

1. 對五個strata進行blind paired listening test。
2. 使用較大型Whisper與第二個quality estimator重算同一500-pair manifest。
3. 評估較低strength或axis-aware calibration，保留interaction reduction並降低axis overshoot。
4. 將Mini 2x、Mini RLACE-8與Large 2x整理成paper ablation。

### 結果來源

- `binding_effect/runs/analysis/parler-mini/large-v1/`
- `binding_effect/runs/analysis/parler-large/constant-steering-v1/`
- `binding_effect/data/shared/DEBIAS_MODEL_COMPARISON.md`
