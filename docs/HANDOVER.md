# Handover note：COMP 4330SEF P04 Group 5 LightRAG

更新日期：2026-11-17。專案位置：`S:\4330 gp`（由 `X:\4330 gp` 搬過嚟，X: 嗰份已經過時，唔好再用）。
GitHub：<https://github.com/keithku321/Exploration-and-Evaluation-of-LightRAG>（private）。

## 1. 而家去到邊

| 項目 | 狀態 |
|---|---|
| Exp 1：naive RAG vs LightRAG（mix） | 本機完成，46 題 |
| Exp 2：chunk 600/1200 × top-k 20/40 | 本機完成（13-Nov），46 題 |
| Exp 3：5 個 query mode | 本機完成，46 題 |
| 改進（proposal 4.7） | **未開始** |
| Colab GPT-4o-mini run | **未做**（proposal 寫嘅模型） |
| 最終報告 + code | **6-Dec-2026 23:59** 交（OLE，zip；報告 ≤2000 字，PDF 名跟指引格式 `SessionPXX_GroupXX Final Report.pdf`，例如 `SessionP04_Group05 Final Report.pdf`，交之前同組長確認個 group 編號點寫） |

本機全部用 qwen2.5:14b（Ollama）答題同做 RAGAS judge，embedding 係 bge-m3。

## 2. 結果喺邊

全部喺 `share/`（已 push）：

- `per_question.csv`：460 行 = 46 題 × 10 個系統，有 answer、hit、nDCG、faithfulness、answer relevancy、tokens、latency。
- `run_info.json`：模型、設定、corpus（20 份文件、95,302 tokens、67,078 words）、index 成本。
- `exp1_summary.csv`、`exp2_summary.csv`、`exp3_summary.csv`：各系統平均。
- `DATA_NOTES.md`：完整結果表同所有要小心講嘅位，寫報告前必讀。

重新生成：`.venv\Scripts\python -X utf8 src\export.py`。原始答案同 index 喺 `results/`（git-ignored，約 70 MB）。

## 3. 重點數字（本機，46 題平均）

| 系統 | Hit | nDCG | Faithfulness | Answer rel. |
|---|---|---|---|---|
| Naive RAG（Exp 1） | 0.957 | 0.821 | 0.899 | 0.834 |
| LightRAG mix（Exp 3 版本） | 0.913 | 0.833 | 0.895 | 0.857 |
| LightRAG naive mode | 0.957 | 0.830 | 0.926 | 0.892 |
| chunk 600 / top-k 40 | 0.804 | 0.748 | 0.925 | 0.828 |
| chunk 1200 / top-k 20 | 0.957 | 0.822 | 0.882 | 0.884 |

Index 成本：chunk 1200 約 161 分鐘、156 次 LLM call、約 1.0M tokens（中途 resume 過，係下限）；chunk 600 約 322 分鐘、405 次 call、約 2.1M tokens。

## 4. 寫報告時一定要講清楚

1. **本機數同 Colab 數唔好混用**，模型唔同。
2. **冇 `judge_correct` 欄**，從來冇跑過正確性評審。最接近係 `token_f1` 同 RAGAS 分數，唔好叫佢做 accuracy。
3. **同一個模型答題又做 judge**，有 bias，要寫入 limitation。
4. **16/46 題抽唔到低層 keyword**，所以 local / hybrid 同其他 mode 嘅分別會細過實際。
5. **Exp 1 嘅 LightRAG faithfulness（0.829）只係用 chunk 文字評**，偏低；Exp 2/3 有計埋 entity/relation 描述。公平比較用 Exp 3 mix（0.895）。
6. **top-k 40 hit rate 反而低過 20**，推測係 12k context 上限俾 entity/relation 食咗位，但**未驗證**，要寫係推測。
7. **n = 46、每個設定只跑一次**，細差距唔可以話係顯著。
8. **Hit / nDCG 係文件層面**，唔係 passage 層面；唔同 chunk size 嘅 nDCG 只可以粗略比較。
9. **問題全部嚟自 UltraDomain `mix.jsonl`**（qasper 19、2WikiMQA 16、multifieldqa 8、HotpotQA 3），冇自己作。評分方法同原論文（LLM win rate）唔同，唔好講成「重現原論文評分」。

## 5. 組員嘅新改動（DeepSeek judge）

組員 Test-Plus 喺 15-Nov push 咗 2 個 commit，18-Nov 已經 pull 落 `S:\4330 gp`：

- 用 DeepSeek（DeepSeek Harness）做第三方 judge，平均 3–5 次。
- 加咗 `results/smoke`、`results/dsh-smoke` 等測試結果，改咗 `share/DATA_NOTES.md`。

因為預設 judge 已經改咗做 DeepSeek，用返本機 qwen judge 嘅分數 export 要咁跑：

```
$env:LIGHTRAG_JUDGE_BACKEND='ollama'; .venv\Scripts\python -X utf8 src\export.py
```

另外組員之前加咗自動 fallback：如果部機冇 qwen2.5:14b，`config.py` 會靜靜雞轉用 qwen3.5:9b，只會喺 stderr 印一行 `[config]`。用 9B 跑出嚟嘅數唔可以同上面嘅比較。

## 6. 下一步（建議次序）

1. 決定最終報告用邊個 judge 嘅分數（本機 qwen judge 定組員嘅 DeepSeek judge）。
2. Colab GPT-4o-mini run（notebook section 10 會出同格式嘅 `per_question.csv` 同 `run_info.json`）。Key 只放 Colab Secrets（`OPENAI_API_KEY`），**之前漏過一次嘅 key 要確認已經 revoke**。
3. 揀一個改進方向做（proposal 4.7：按問題類型調 low/high-level 權重、改 extraction prompt、加 reranker）。Exp 2 結果顯示 context budget 分配可能係一個切入點。
4. 報告（≤2000 字，英文）：Abstract 100–150 字、Intro、Lit review、Methodology、Results、Conclusion、References、Appendix（work allocation）。

## 7. 點樣跑（本機）

```
.venv\Scripts\python -X utf8 src\exp1.py index | run
.venv\Scripts\python -X utf8 src\evaluate.py results\exp1_naive.jsonl results\exp1_lightrag.jsonl
.venv\Scripts\python -X utf8 src\exp2.py index | run | score
.venv\Scripts\python -X utf8 src\exp3.py run | score
.venv\Scripts\python -X utf8 src\export.py
```

全部步驟都可以中斷後接住跑。唔好同時跑 index 同評分，亦唔好同時用 GPU 跑其他嘢（例如 HKFirstAIVtuber），會爭 GPU。
