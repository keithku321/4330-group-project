# COMP 4330SEF P04 Group 5: 剩餘工作清單（按時間順序）

題目：Exploration and Evaluation of LightRAG
Tutorial：P04（逢星期一 17:00–17:50，C0412）

## 重要日期

| 日期 | 事項 |
|---|---|
| **15/11（日）23:59** | 初稿 slides 上傳 OLE（之後仍可修改） |
| **16/11（一）17:00** | Presentation：6 分鐘 + 2–3 分鐘 Q&A，**全部組員都要到場** |
| **4/12（四）** | 組內目標：final report + code 完成 |
| **6/12（日）23:59** | OLE 硬性截止，**不接受遲交** |

最後一天前（5/12）仍可替換已交嘅版本。

## 已完成

- [x] Proposal
- [x] 本機 Exp 1：Naive RAG vs LightRAG（qwen2.5:14b + bge-m3），結果已寫入 notebook
- [x] 評分流程：hit rate、nDCG、RAGAS faithfulness / answer relevancy、token F1、cost
- [x] Colab notebook：`COMP4330_Group5_LightRAG.ipynb`
- [x] GitHub private repo：`keithku321/comp4330-lightrag`（notebook 同 README 更新後未 push）

本機結果摘要（n = 46）：

| | Naive RAG | LightRAG (mix) |
|---|---|---|
| Hit rate | 0.957 | 0.913 |
| nDCG | 0.821 | 0.833 |
| Faithfulness | 0.899 | 0.829 |
| Answer relevancy | 0.834 | 0.859 |
| Token F1 | 0.403 | 0.358 |
| Prompt tokens / 條 | 9,267 | 12,206 |
| Latency (s) | 10.9 | 18.1 |

LightRAG index 成本：161 分鐘、156 次 LLM call、約 100 萬 tokens。

---

## 第一階段：Presentation 前（10/11 – 16/11）

### 1. 全組對一對結果（今日至 11/11）
- [ ] 全組睇一次上面嘅結果，確認大家明白「LightRAG 今次冇贏」
- [ ] 決定 presentation 嘅主線：**誠實報告結果，再分析原因**，老師想聽 analysis，唔係介紹別人嘅 work
- [ ] 同 tutor 確認難度級別（Medium 定 High）；hit rate / nDCG 係文件層面，同 tutor 講明
- [ ] 決定邊個負責講、邊個負責 Q&A（全部人都要在場）

### 2. 在 Colab 用 Gemini API 跑一次（proposal 寫 GPT-4o-mini，報告要講明改用 Gemini 同原因）
- [ ] 上傳 `COMP4330_Group5_LightRAG.ipynb` 去 Colab
- [ ] 左邊 🔑 Secrets 加 `GEMINI_API_KEY`（`OPENAI_API_KEY` 呢個名都得），開啟 notebook access，Runtime 揀 **CPU**
- [ ] 先將 `QUESTION_LIMIT = 5` 試水，確認 key 同流程冇問題
- [ ] 改返 `QUESTION_LIMIT = None`，Run all，等結果
- [ ] 記低 Gemini 實際用咗幾多錢（作為 cost 數據）
- [ ] 留意：Colab 結果同本機結果係**兩套數字**（LLM 唔同），報告要分開寫，唔好混埋

### 3. Exp 3：五個 query mode 比較（naive / local / global / hybrid / mix）
Notebook 第 9 節已經寫好（接喺 Exp 1 評分之後），**唔使另外寫 code**，Run all 就會跑。同一個 index，只轉 `mode`，所以成本低。`mix` 直接沿用 Exp 1 嘅 LightRAG 答案，唔會重複計錢。

Notebook 會自動做以下嘢：
1. **Keyword 診斷**：檢查 LLM 抽出嘅低層 / 高層 keyword 有冇空。
2. 用 `naive`、`local`、`global`、`hybrid` 四個 mode 各答一次所有問題（`mix` 沿用 Exp 1）。
3. 評分（同 Exp 1 一樣嘅 metrics），再出一張 mode 比較表，另存 `exp3_summary.csv`。

你要做：
- [ ] Colab 上 Run all，等第 9 節跑完
- [ ] **先睇 keyword 診斷結果，再解讀比較表**。本機 qwen2.5:14b 有 16/46 條問題嘅低層 keyword 係空，咁 `local` 會跌入 hybrid 路徑，同 `hybrid` 一樣。表入面如果 `n_entities` / `n_relations` 完全相同，就代表 mode 之間冇真正分別，請照實報告，唔好硬排名
- [ ] 比較各 mode 嘅質素同 token 消耗，揀出最好嘅 mode，作為改進方向嘅依據
- [ ] 如果想連本機 qwen 版本都有 Exp 3 數字，同我講，我可以喺本機跑（約 1 個鐘）

### 3b. 用 Colab 數據做 report / slides：注意事項
最終數字一律來自 Colab（Gemini）。本機 qwen 數據只作備份同附錄，**兩套唔好混用**。
- [ ] 全部數字由 Drive 檔案取：`MyDrive/COMP4330_LightRAG/results_gemini/`，唔好用截圖抄數字
- [ ] 全跑完（`QUESTION_LIMIT = None`）先執行第 10 節匯出，得到 `per_question.csv` 同 `run_info.json`
- [ ] 跑完立刻另存一份有輸出嘅 notebook：File → Save a copy in Drive，同下載 `.ipynb`；Colab 關閉後 runtime 內嘅輸出會消失
- [ ] 記低 Google AI Studio 後台嘅 Usage嘅實際費用，同 index 成本互相核對
- [ ] 試跑（`QUESTION_LIMIT = 5`）同全跑要用**同一套設定**，中途唔好改 `TARGET_TOKENS`、chunk size，否則結果檔會混入舊數據；要重新開始就刪走 `results_gemini` 資料夾
- [ ] 本機 Exp 1 嘅 LightRAG faithfulness 只用 chunk 文字評分，Colab 版本已加入 entity / relation 描述，兩者不可直接比較
- [ ] Gemini 同時負責答題同評分，bias 仍然存在，要寫入 limitation
- [ ] 只有 Colab 版本嘅 hit rate / nDCG 仍然係文件層面

### 4. 準備 slides（6 分鐘）
建議結構同時間：

| 時間 | 內容 |
|---|---|
| 0:00–1:00 | 問題：naive RAG 點解 fail multi-hop；LightRAG 係咩 |
| 1:00–1:30 | 實驗設定（一張圖：corpus、問題、LLM、metrics） |
| 1:30–4:30 | **結果同分析（重點）**：Exp 1、Exp 3、點解 LightRAG 今次冇贏 |
| 4:30–5:30 | 下一步：Exp 2 同改進方向 |
| 5:30–6:00 | 結論同 limitation |

- [ ] 揀 2–3 張圖：Exp 1 比較表、cost 對比（index 成本 + query token）、Exp 3 mode 比較
- [ ] 分析原因要講：corpus 細（95k tokens）、只有 19/46 係 multi-hop、extraction 輸出被 3000 token 上限截斷、judge 同答題係同一個 model
- [ ] Live demo 最多 30 秒，或者用錄好嘅 screen recording，唔好依賴現場 Colab
- [ ] 全組 slides 排版統一，由一個人整合

### 5. 交初稿 slides
- [ ] **15/11 23:59 前**上傳 OLE（可以之後再改）

### 6. Rehearsal 同 Q&A 準備
- [ ] 用計時器全組練至少 2 次，必須 ≤ 6 分鐘
- [ ] 準備預計問題：
  - 點解 LightRAG 輸咗？
  - 46 條題、跑一次，結果可唔可以信？
  - Judge 同答題係同一個 model，公唔公平？
  - 點解用 qwen2.5:14b 而唔係 Gemini API？
  - 你哋打算點樣改進？
  - hit rate / nDCG 點樣計？

### 7. Present
- [ ] **16/11（一）17:00**，C0412，全組準時到場

---

## 第二階段：Presentation 後（17/11 – 4/12）

### 8. Exp 2：chunk size 同 top-k
- [ ] 縮細成 2×2（例如 chunk 600 / 1200 × top-k 20 / 40）
- [ ] 注意：每改一次 chunk size 都要**重新建 index**，所以呢個最貴
- [ ] 用同一個問題集同 metrics

### 9. 改進（揀一個）
- [ ] 揀其中一個方向，建議揀唔使重新建 index 嘅：
  - 加 reranker
  - 按問題類型調整 low-level / high-level 檢索權重
- [ ] 如果揀「調整 entity extraction prompt」，要重新建 index（昂貴），但最貼合今次觀察到嘅問題（輸出被截斷）
- [ ] 對比改進前後嘅數字

### 10. 補做 retrieval 評估
- [ ] 現時 hit rate / nDCG 係文件層面。如果要更精細，需要自己標註一小批 gold passage，或者用 RAGAS 嘅 context precision / recall 代替
- [ ] 同 tutor 或組員決定做唔做

### 11. 寫 final report（≤ 2,000 字，唔計參考文獻、圖表、表格內文字、work allocation、appendix）
必須包括，按次序：
1. Title
2. **Abstract**（100–150 字）
3. Introduction：Background 同 problem definition
4. Literature Review
5. Methodology：對被研究系統嘅詳細分析
6. Results / Evaluation：評估同提出可能嘅改進
7. Conclusion
8. References
9. Appendix：work allocation，及其他必要內容

評分比重（report 佔 project 30%）：Abstract 10%、Introduction + Literature Review 20%、Methodology + Results 60%（乘難度系數）、Conclusion + References 10%。

- [ ] 用 English 寫（report 同 presentation 都必須英文）
- [ ] 全組分工寫，組長整合

### 12. 檢查 proposal 帶落嚟嘅問題
- [ ] **Reference 混淆**：Literature review 3.1 嘅 "Gao et al. (2023)" 係 survey（arXiv:2312.10997），但 reference list 亦有另一篇 Gao et al. 2023（HyDE）。兩篇要分清楚
- [ ] 核對 LightRAG 原 paper 嘅數字：「84.8% win rate」同「每次 retrieval 少過 100 tokens」
- [ ] 檢查所有 reference 格式一致

### 13. 必須在 report 交代嘅 limitation
- [ ] 本機用 qwen2.5:14b，Colab 用 Gemini，兩套結果唔同
- [ ] Context 由 LightRAG 預設 30k 減到 16k
- [ ] Entity extraction 輸出上限 3000 tokens，多數 chunk 被截斷
- [ ] 只用咗 20 篇文件（95k tokens）、46 條問題、跑一次
- [ ] Judge 同答題用同一個 LLM
- [ ] hit rate / nDCG 係文件層面
- [ ] 關閉咗 reranker
- [ ] 本機 qwen2.5:14b 有 16/46 條問題抽唔到低層 keyword，實體層檢索冇起作用
- [ ] 本機 Exp 1 嘅 LightRAG faithfulness 只用 chunk 文字做 judge context，冇包括 entity / relation 描述，數字可能偏低；Colab 版本已修正

### 14. 整理 code 同提交
- [ ] 更新 `README.md`，確保任何人都可以重現
- [ ] Push 去 GitHub（同我講我先 push）
- [ ] 如果 code 太大，用雲端連結
- [ ] 檢查 repo 入面冇 API key
- [ ] Report 轉做**單一 PDF**，檔名：`SessionP04_Group5 Final Report.pdf`
- [ ] 將 report、slides、code 打包做 zip
- [ ] **組長**一個人上傳 OLE（只需交一份）
- [ ] 目標 **4/12** 完成，**6/12 23:59** 係死線，唔准遲交
- [ ] 提交前跑 plagiarism 自我檢查（Turnitin 會檢查）

---

## 分工（來自 proposal）

| 成員 | 負責 |
|---|---|
| Chen Zijun（組長） | 協調、聯絡 tutor、proposal、final report |
| Wang Sibo | 環境設定、LightRAG 安裝同 demo |
| Ho Kwun Chi、Chan Chi Kin | Naive RAG、Exp 1 |
| Poon Ho Man | Exp 2、Exp 3、結果分析 |
| Ku Sui Ki | 評估指標、RAGAS judge、表格同圖 |
| 全組 | Slides、6 分鐘 presentation、Q&A 練習 |

## 組隊注意事項

- 每位組員都要參與；唔參與嘅組員可能會被視為 non-participant，甚至零分
- 如果要分數唔平均，所有組員同意後，由組長用 email 提交 share percentage
- 組員有責任喺一個禮拜內回覆組長嘅 email
- 組長如果要報告 non-participant，需要至少兩次聯絡嘗試嘅證據，隨作業一併提交
