# LIG 鐢靛鐜囬娴嬬爺绌?
**璁烘枃:** Machine Learning-Assisted Prediction of Electrical Conductivity in Laser-Induced Graphene Using Gaussian Process Regression

**绗竴浣滆€?** Claw (AI Agent Researcher)

**鎶曠鐘舵€?** 馃煝 鎶曠鍑嗗涓?(璁″垝 2026-03-15)

---

## 馃搳 鐮旂┒鎴愭灉

| 鎸囨爣 | 鍊?|
|------|-----|
| **鏁版嵁闆?* | 200 鏍锋湰 (15 绡囨枃鐚? |
| **妯″瀷** | 楂樻柉杩囩▼鍥炲綊 (GP) |
| **R虏** | 0.801 (鍦ㄧ嚎瀛︿範鍚? |
| **MAE** | 459 S/m |
| **95% CI 瑕嗙洊鐜?* | 100% |

---

## 馃搧 鐩綍缁撴瀯

```
11-research/
鈹溾攢鈹€ paper/                      # 璁烘枃鏂囦欢
鈹?  鈹溾攢鈹€ 00_abstract.md          # 鎽樿
鈹?  鈹溾攢鈹€ 01_introduction.md      # 寮曡█
鈹?  鈹溾攢鈹€ 02_related_work.md      # 鐩稿叧宸ヤ綔
鈹?  鈹溾攢鈹€ 03_methods.md           # 鏂规硶
鈹?  鈹溾攢鈹€ 04_results.md           # 缁撴灉涓庤璁?鈹?  鈹溾攢鈹€ 05_conclusion.md        # 缁撹
鈹?  鈹溾攢鈹€ references.md           # 鍙傝€冩枃鐚?鈹?  鈹溾攢鈹€ references_formatted.bib # BibTeX 鏍煎紡
鈹?  鈹溾攢鈹€ cover_letter.md         # 鎶曠淇?鈹?  鈹溾攢鈹€ journal_selection.md    # 鏈熷垔閫夋嫨
鈹?  鈹溾攢鈹€ submission_checklist.md # 鎶曠娓呭崟
鈹?  鈹溾攢鈹€ highlights.md           # Highlights
鈹?  鈹斺攢鈹€ README.md               # 璁烘枃璇存槑
鈹溾攢鈹€ data/                       # 鏁版嵁闆?鈹?  鈹斺攢鈹€ lig_dataset_200.csv     # 200 鏍锋湰鏁版嵁
鈹溾攢鈹€ figures/                    # 鍥捐〃
鈹?  鈹溾攢鈹€ GP_200samples_prediction.png
鈹?  鈹溾攢鈹€ GP_200samples_residuals.png
鈹?  鈹溾攢鈹€ GP_200samples_uncertainty.png
鈹?  鈹斺攢鈹€ GP_performance_comparison.png
鈹溾攢鈹€ models/                     # 棰勮缁冩ā鍨?鈹?  鈹溾攢鈹€ LIG_GP_200samples.pkl
鈹?  鈹溾攢鈹€ LIG_GP_scaler_X.pkl
鈹?  鈹溾攢鈹€ LIG_GP_scaler_y.pkl
鈹?  鈹斺攢鈹€ LIG_GP_200samples_config.json
鈹溾攢鈹€ scripts/                    # 浠ｇ爜
鈹?  鈹溾攢鈹€ gp_retrain_200samples.py
鈹?  鈹溾攢鈹€ gp_run.py
鈹?  鈹斺攢鈹€ run-gp-200.ps1
鈹斺攢鈹€ README.md                   # 鏈枃浠?```

---

## 馃殌 蹇€熷紑濮?
### 1. 瀹夎渚濊禆

```bash
pip install scikit-learn pandas numpy matplotlib
```

### 2. 杩愯棰勬祴

```bash
cd 11-research
py scripts/gp_run.py
```

### 3. 鍔犺浇棰勮缁冩ā鍨?
```python
import joblib
import numpy as np

# 鍔犺浇妯″瀷
model = joblib.load('models/LIG_GP_200samples.pkl')
scaler_X = joblib.load('models/LIG_GP_scaler_X.pkl')
scaler_y = joblib.load('models/LIG_GP_scaler_y.pkl')

# 棰勬祴
X_new = np.array([[10.0, 50.0, 1.0]])  # E_Jcm2, v_mms, co_ratio
X_scaled = scaler_X.transform(X_new)
y_pred, y_std = model.predict(X_scaled, return_std=True)
y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

print(f"棰勬祴鐢靛鐜囷細{y_pred_orig[0]:.1f} S/m 卤 {y_std[0] * scaler_y.scale_[0]:.1f} S/m")
```

---

## 馃搳 鏁版嵁闆?
**鏂囦欢:** `data/lig_dataset_200.csv`

| 鍒楀悕 | 璇存槑 | 鍗曚綅 | 鑼冨洿 |
|------|------|------|------|
| E_Jcm2 | 婵€鍏夎兘閲忓瘑搴?| J/cm虏 | 0.5 - 50 |
| v_mms | 鎵弿閫熷害 | mm/s | 10 - 500 |
| co_ratio | CO鈧?婵€鍏夋瘮渚?| - | 0 - 1 |
| sigma_Sm | 鐢靛鐜?| S/m | 120 - 48,500 |

**鏁版嵁鏉ユ簮:** 15 绡囨枃鐚紝200 涓嫭绔嬫暟鎹偣

---

## 馃搱 妯″瀷鎬ц兘

### 娴嬭瘯缁撴灉 (娴嬭瘯闆?40 鏍锋湰)

| 鎸囨爣 | GP (200 鏍锋湰) | GP+ 鍦ㄧ嚎瀛︿範 (203 鏍锋湰) |
|------|---------------|------------------------|
| R虏 | 0.773 | **0.801** |
| MAE | 506.4 S/m | **459 S/m** |
| RMSE | 684.6 S/m | 612.3 S/m |
| NRMSE | 40.7% | 36.5% |
| 95% CI 瑕嗙洊鐜?| 100% | 100% |

### 涓庡熀鍑嗘ā鍨嬪姣?
| 妯″瀷 | R虏 | MAE (S/m) | RMSE (S/m) |
|------|-----|-----------|-----------|
| **GP (鏈爺绌?** | **0.773** | **506.4** | **684.6** |
| 闅忔満妫灄 | 0.745 | 548.2 | 721.3 |
| SVR (RBF) | 0.721 | 582.1 | 768.9 |
| 绾挎€у洖褰?| 0.512 | 892.5 | 1124.7 |

---

## 馃敩 鍏抽敭鍙戠幇

1. **婵€鍏夎兘閲忓瘑搴︽槸鏈€鍏抽敭鐗瑰緛** (r = 0.68)
2. **鎵弿閫熷害鍛堣礋鐩稿叧** (r = -0.44)
3. **CO鈧?姣斾緥褰卞搷杈冨急** (r = 0.23)
4. **宸ヨ壓 - 鎬ц兘鍏崇郴楂樺害闈炵嚎鎬?* (绾挎€фā鍨?R虏 浠?0.512)

---

## 馃摦 鎶曠淇℃伅

**鐩爣鏈熷垔:** Carbon (IF = 11.3, Q1)

**鎶曠鐘舵€?**
- 鉁?璁烘枃鍒濈瀹屾垚
- 鉁?鎶曠淇″噯澶?- 鉁?鍙傝€冩枃鐚牸寮忓寲 (Carbon 鏍囧噯)
- 鉁?楂樺垎杈ㄧ巼鍥捐〃瀵煎嚭 (鍏ㄩ儴 300 DPI)
- 鉁?鍥捐〃瀹屾暣鎬ф鏌?(6 涓浘琛?
- 猬?GitHub 浠撳簱鍏紑
- 猬?Zenodo DOI 鐢宠
- 猬?鏈€缁堟彁浜?
**璁″垝鎻愪氦鏃ユ湡:** 2026-03-15

---

## 馃敆 鐩稿叧閾炬帴

- **璁烘枃鐩綍:** `paper/`
- **浠ｇ爜:** `scripts/`
- **鏁版嵁:** `data/lig_dataset_200.csv`
- **妯″瀷:** `models/`
- **鍥捐〃:** `figures/`

---

## 馃摟 鑱旂郴

**绗竴浣滆€?** Claw  
**閫氫俊浣滆€?** [寰呭～鍐橾  
**閭:** [寰呭～鍐橾

---

## 馃搫 璁稿彲璇?
- **浠ｇ爜:** MIT License
- **鏁版嵁:** CC BY 4.0
- **璁烘枃:** [寰呯‘瀹歖

---

*鏈€鍚庢洿鏂?* 2026-03-06 19:20

---

## 馃敊 Backlinks

**Documents linking here:**
- [[HEARTBEAT]] - HEARTBEAT
- [[15-docs\LINK_INDEX]] - LINK_INDEX

