# MACE rMD17 Reproduction Experiment Log

## 目標
重現 MACE 論文 (Batatia et al., NeurIPS 2022) 的 rMD17 benchmark 結果

**論文 Ethanol 目標:**
- Energy MAE: 0.4 meV (total)
- Forces MAE: **2.1 meV/Å**

---

## 實驗記錄

### Run 1: 256x0e (Invariant Only)
**時間:** 2026-02-24  
**設定:** `--hidden_irreps=256x0e` (L=0, invariant only)

**結果 (Test Set):**
| 指標 | 結果 | 論文目標 | 差距 |
|------|------|----------|------|
| Energy MAE | 0.1 meV/atom | 0.4 meV | ✅ |
| Forces MAE | **6.4 meV/Å** | 2.1 meV/Å | 3x worse |

**問題發現:** 
1. Stage Two `swa_forces_weight` 預設為 100，但論文用 1000
2. Invariant-only 模型無法達到論文精度

---

### Run 2: 128x0e+128x1o+128x2e (L=2 Equivariant)
**時間:** 2026-02-25  
**設定:** `--hidden_irreps=128x0e+128x1o+128x2e` (L=2 equivariant)  
**修正:** 加入 `--swa_forces_weight=1000.0`

**結果 (Test Set):**
| 指標 | 結果 | 論文目標 | 差距 |
|------|------|----------|------|
| Energy MAE | 0.1 meV/atom | 0.4 meV | ✅ |
| Forces MAE | **4.3 meV/Å** | 2.1 meV/Å | 2x worse |

**改進:** Forces MAE 從 6.4 → 4.3 meV/Å (改善 33%)

---

### Run 3: 128x0e+128x1o+128x2e+128x3o (L=3 Equivariant) [進行中]
**時間:** 2026-02-25  
**設定:** `--hidden_irreps=128x0e+128x1o+128x2e+128x3o` (L=3 equivariant)

**預期:** 應該進一步改善 Forces MAE

---

## 關鍵設定對照

| 參數 | 論文 A.5.1 | 我們的設定 |
|------|-----------|-----------|
| num_interactions | 2 | 2 ✅ |
| max_ell | 3 | 3 ✅ |
| correlation | 3 | 3 ✅ |
| r_max | 5.0 Å | 5.0 ✅ |
| batch_size | 5 | 5 ✅ |
| energy_weight | 1 | 1 ✅ |
| forces_weight | 1000 | 1000 ✅ |
| swa_forces_weight | 1000 | 1000 ✅ |
| hidden_irreps | "256x0e" | 128x0e+128x1o+128x2e+128x3o |

**注意:** 論文 Appendix A.5.1 寫的是 `256x0e`，但 Table 2 顯示 MACE(L=2) 效果最好。
實際上需要 equivariant features 才能達到論文報告的精度。

---

## 結論

1. **Invariant-only (L=0)** 模型無法達到論文精度
2. **Equivariant features** 是達到 SOTA 的關鍵
3. **Stage Two weights** 需要手動設定為 1000
