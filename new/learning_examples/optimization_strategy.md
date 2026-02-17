# Phase 1 訓練優化策略：理論基礎與學術解析 (Theoretical Foundations of Optimization Strategy)

本文件旨在從數學原理與機器學習理論的角度，深入解析 `scripts/11_train_cgcnn_full.py` 中採用的優化策略。這些策略的選擇並非經驗法則 (Example-based)，而是基於最新的深度學習研究文獻，旨在解決圖神經網路 (GNN) 在非凸優化 (Non-convex Optimization) 中的收斂難題。

---

## 1. 學習率調度：One Cycle Policy 與超收斂現象

### 理論背景 (Theoretical Background)
傳統的學習率衰減策略 (如 Step Decay, Exponential Decay) 假設模型在訓練初期即處於強凸區域附近，僅需逐步減小步伐以收斂至極小值。然而，Leslie N. Smith 在 *A discipline of neural network hyper-parameters* (2018) [1] 中指出，深度神經網路的損失地形 (Loss Landscape) 充滿了鞍點 (Saddle Points) 與尖銳的局部極小值 (Sharp Minima)。

### 我們的實作：One Cycle Policy
我們採用 Smith 提出的 **One Cycle Policy**，其學習率 $\eta_t$ 隨訓練步數 $t$ 的變化遵循以下兩個階段：

1.  **Warm-up Phase (0% - 30%)**：
    $$ \eta_t = \eta_{min} + (\eta_{max} - \eta_{min}) \cdot \frac{t}{T_{warm}} $$
    學習率線性攀升至峰值。
    *   **學術意義**：高學習率使模型能夠穿越損失地形中的鞍點，並具有「正則化」效果，強迫模型跳出狹窄的局部極小值 (這些極小值通常泛化能力較差)，尋找更寬闊平坦的極小值 (Flat Minima) [2]。

2.  **Annealing Phase (30% - 100%)**：
    學習率由 $\eta_{max}$ 緩慢衰減至 $\eta_{min}/1000$ (甚至更低)。
    *   **學術意義**：在進入平坦區域後，微小的學習率允許模型進行精細的參數微調，以達到極致的收斂精度 (Super-convergence)。

與此同時，動量 (Momentum $\beta_t$) 採用 **Inverse Triangular** 策略（先降後升），這與學習率的變化相反，進一步穩定了高學習率時期的參數更新方向。

---

## 2. 梯度控制：Gradient Clipping (梯度剪切)

### 數學原理 (Mathematical Principle)
在深層 GNN 或 RNN 中，梯度是通過鏈式法則 (Chain Rule) 反向傳播的。假設網路層數為 $L$，權重矩陣為 $W$，則梯度 $\nabla L$ 與 $\prod_{i=1}^{L} W_i$ 成正比。
*   如果 $W_i$ 的最大特徵值 $\lambda > 1$，則梯度隨層數呈指數級增長 (Exploding Gradients)，導致權重更新步長過大，破壞已學習的特徵。
*   這在 Pascanu et al. (2013) 的論文 [3] 中有詳盡的數學證明。

### 我們的實作：Norm Clipping
我們對梯度的 $L_2$ 範數 (Norm) 實施硬約束：
$$ g \leftarrow \frac{g \cdot \text{max\_norm}}{\max(\|g\|_2, \text{max\_norm})} $$
其中 $\text{max\_norm} = 0.5$。
這確保了梯度的方向 (Direction) 不變，僅限制其量級 (Magnitude)。這對於 GNN 尤為重要，因為 GNN 的消息傳遞機制 (Message Passing) 類似於 RNN，極易產生數值不穩定。

---

## 3. 損失函數：Huber Loss (魯棒回歸)

### 數學定義 (Mathematical Definition)
MSE (均方誤差) 對異常值 (Outliers) 極度敏感，因為誤差 $e$ 被平方放大 ($e^2$)。MAE (平均絕對誤差) 雖然魯棒，但在 $e=0$ 處不可微，導致優化困難。
Huber Loss [4] 結合了兩者的優點：

$$
L_{\delta}(y, f(x)) = 
\begin{cases} 
\frac{1}{2}(y - f(x))^2 & \text{for } |y - f(x)| \le \delta \\
\delta (|y - f(x)| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

### 我們的實作：$\delta = 0.2$
我們將 $\delta$ 設定為 $0.2$ eV/atom。
*   **物理意義**：這意味著對於誤差小於 $0.2$ eV 的預測，我們認為它已經接近真實值，使用 $L2$ Loss 加速收斂。對於誤差大於 $0.2$ eV 的預測 (可能是數據噪聲或難以預測的樣本)，我們使用 $L1$ Loss 進行線性懲罰，避免模型為了迎合這些離群點而大幅扭曲參數空間。

---

## 4. 數據預處理：Statistical Outlier Removal

### 統計原理
基於常態分佈假設 (Gaussian Assumption)，數據點落在平均值 $\mu$ 之外 $k$ 個標準差 $\sigma$ 的概率由高斯積分給出。
*   對於 $k=4$ ($4\sigma$)，$P(|x-\mu| > 4\sigma) \approx 6.3 \times 10^{-5}$。
*   換言之，一個合法的數據點出現在 $4\sigma$ 之外的機率極低。在材料科學數據集中，這類點通常代表計算失敗 (Convergence Failure) 或非物理結構。

### 我們的實作
將過濾門檻從 $10\sigma$ 降至 $4\sigma$ 是基於 **Robust Statistics** 的原則。這雖然移除了約 0.1% 的數據，但顯著降低了訓練數據的峰度 (Kurtosis)，使損失地形更加平滑，從而允許使用更大的學習率進行優化。

---

## 5. 數據爆炸的防禦機制：縱深防禦 (Defense in Depth)

**(特別增補：如何解決梯度爆炸與 NaN 問題)**

在先前的失敗嘗試中，訓練過程因 `Loss = NaN` 而崩潰。這通常是因為梯度在反向傳播過程中數值溢出 (Overview flow) 或極端數據導致的。我們採取了三道防線來「拉住」即將失控的模型：

### 第一道防線：源頭阻絕 (Outlier Removal)
*   **問題**：DFT 計算可能有誤，產生如 2000 eV 的異常值。這些值在計算 Loss 時會產生天文數字般的梯度。
*   **解法**：**4-Sigma 過濾**。
    *   在數據進入模型前，直接剔除這些「地雷」。
    *   就像在機場安檢時就攔下爆裂物，不讓它上飛機。

### 第二道防線：傷害控制 (Huber Loss)
*   **問題**：即使過濾後，仍可能有誤差較大的樣本 (例如預測 -1.0，實際 -5.0)。如果用 MSE $(e^2)$，誤差會被放大成 16 倍的梯度。
*   **解法**：**Huber Loss (Linear Tail)**。
    *   對於大誤差，Loss 函數會自動切換成線性模式 (Linear Mode)。
    *   這意味著梯度是常數，不會因為誤差變大而無限膨脹。這限制了單個樣本能對模型造成的最大「傷害」。

### 第三道防線：強制制動 (Gradient Clipping)
*   **問題**：GNN 的多層結構會導致梯度連乘。即使單個樣本沒問題，經過 5 層卷積後，梯度可能累積到非常大。
*   **解法**：**Gradient Norm Clipping = 0.5**。
    *   這是最後的保險絲。
    *   在更新權重前，我們檢查整個網路的梯度長度。
    *   **如果梯度長度 > 0.5，我們就強制把它縮小** (例如除以 100)，保持方向不變，但步伐變小。
    *   這保證了無論模型遇到多麼崎嶇的地形，參數更新的步幅永遠被限制在安全範圍內 (0.5)，絕對不會一步踏空 (變成 NaN)。

---

## 6. 優化策略的局限與潛在風險 (Limitations and Trade-offs)

儘管目前的優化策略取得了顯著的成功，但我們必須客觀地認識到它的邊界條件：

### OneCycleLR 的剛性限制 (OneCycleLR Rigidity)
*   **斷點續訓困難 (Resume Difficulty)**：OneCycleLR 是一個完整的週期 (Cycle)。如果訓練中途意外中斷 (例如第 800 Epoch)，我們很難從中間簡單地「接續」它，因為學習率和動量必須嚴格按照原定的曲線軌跡前進。
*   **超參數敏感 (Hyperparameter Sensitivity)**：如果要改變 Epochs 數量 (例如延長到 3000)，整個學習率曲線必須重新計算，不能像 StepLR 那樣隨意增加。

### Gradient Clipping 的治標性質 (Clipping Masks Instability)
*   **掩蓋深層問題**：梯度剪切雖然能防止爆炸，但它並沒有解決「為什麼梯度會爆炸」的根本原因 (可能是模型架構設計不良或初始化錯誤)。
*   **收斂變慢**：如果 `max_norm` 設得太小 (例如 0.01)，模型更新過於謹慎，可能會在平坦極小值附近徘徊太久，無法快速切入最優解。

### Outlier Removal 的誤殺風險 (False Positives)
*   **可能錯失科學發現 (The Black Swan Problem)**：在材料科學中，某些極端值 (Outlier) 可能真的是物理上的重大發現 (例如室溫超導體的某個前兆)。我們目前的 $4\sigma$ 過濾策略，雖然有利於訓練一般化模型，但也意味著模型失去了對這些「特異點」的學習能力。

### Huber Loss 的收斂精度代價
*   **線性區間**：在高誤差區間，Huber Loss 是線性的 (L1)。這意味著即使誤差還很大，梯度的回饋力度也不會增強。對於某些需要強梯度推動的時刻，這可能會導致收斂速度略慢於 MSE。

---

## 參考文獻 (References)

1.  Smith, L. N. (2018). *A discipline of neural network hyper-parameters: Part 1 - learning rate, batch size, momentum, and weight decay*. arXiv preprint arXiv:1803.09820.
2.  Keskar, N. S., et al. (2016). *On large-batch training for deep learning: Generalization gap and sharp minima*. ICLR 2017.
3.  Pascanu, R., Mikolov, T., & Bengio, Y. (2013). *On the difficulty of training recurrent neural networks*. ICML 2013.
4.  Huber, P. J. (1964). *Robust estimation of a location parameter*. Annals of Mathematical Statistics.
