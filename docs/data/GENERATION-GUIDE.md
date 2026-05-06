# ĐẶC TẢ THUẬT TOÁN SINH DỮ LIỆU MOCK (DATA GENERATOR SPEC)

## 1. THÔNG SỐ TOÀN CỤC (GLOBAL PARAMETERS)
*   **Vector Thời gian:** $t =[0, 1, 2, ..., 239]$ (Tương đương 240 steps).
*   **Phân vùng Giai đoạn (Phase Boundaries):**
    *   `STARTUP_ZONE`: $t \in[0, 35]$
    *   `STABLE_ZONE`: $t \in[36, 204]$
    *   `SHUTDOWN_ZONE`: $t \in[205, 239]$
*   **Hàm tạo nhiễu nền (Background Noise):** $N(\mu, \sigma)$ sử dụng `np.random.normal(0, sigma)`.

---

## 2. THUẬT TOÁN SINH DỮ LIỆU CHUẨN (GOLDEN BASELINE MODELING)
Mỗi cảm biến được mô phỏng bằng một phương trình toán học riêng biệt để tạo ra tín hiệu $X_{norm}(t)$.

### 2.1. `air_motor_speed` (Hình bậc thang - Staircase)
*   **Toán học:** Hàm bước nhảy (Step function) kết hợp với bộ lọc làm mượt (Low-pass filter).
*   **Cách sinh:** 
    1. Khởi tạo mảng 0. 
    2. Gán các mức ngẫu nhiên (ví dụ: 1000, 2500, 1500 RPM) cho các đoạn thời gian ngẫu nhiên trong `STABLE_ZONE`.
    3. Dùng `scipy.ndimage.gaussian_filter1d` để làm vát các góc vuông (tạo độ dốc / Slew rate tự nhiên).
    4. Cộng nhiễu nền: $\sigma = 2.0$.

### 2.2. `electro_pneumatic_output` (Nhiễu tần số cao - High-frequency Ripple)
*   **Toán học:** Xung Purge ban đầu + Dao động ngẫu nhiên biên độ lớn.
*   **Cách sinh:**
    1. $t \in[2, 8]$: Gán giá trị xung cực đại $= 4000$.
    2. $t > 8$: Gán mức cơ bản $= 500$ + Nhiễu rung cực mạnh $\sigma = 150.0$.

### 2.3. `hv_current` (Đường cong Tanh - Sigmoid Ramp-up)
*   **Toán học:** Hàm Sigmoid/Tanh để mô phỏng sự tăng dòng điện từ từ.
*   **Cách sinh:**
    1. Dùng hàm $\tanh(t - t_0)$ để tăng giá trị từ 0 lên 85 $\mu A$ tại `STARTUP_ZONE`.
    2. Cộng nhiễu nền: $\sigma = 3.0$.

### 2.4. `hv_voltage` (Xung vuông âm - Negative Square Pulse)
*   **Toán học:** Hàm xung vuông đảo ngược (Inverted Square Wave).
*   **Cách sinh:**
    1. Gán giá trị $= 0$ tại STARTUP và SHUTDOWN.
    2. Gán giá trị $= -90.0$ tại STABLE.
    3. Làm mượt nhẹ cạnh lên/xuống (Gaussian filter window=3).
    4. Nhiễu nền cực nhỏ: $\sigma = 0.5$ (Để bắt được lỗi Insulation Breakdown).

### 2.5. `paint_pressure` & `paint_flow_rate` (Đồng bộ - Correlated Square Waves)
*   **Toán học:** Hàm xung vuông có chu kỳ (On/Off Spraying).
*   **Cách sinh:**
    1. Tạo một `spray_mask` (Mảng boolean 0 và 1) bằng `scipy.signal.square`.
    2. `flow_rate` $= spray\_mask \times 300$ + nhiễu $\sigma = 5.0$.
    3. `pressure` $= spray\_mask \times 0.2$ + nhiễu hình sin (bơm màng) $0.02 \cdot \sin(\omega t)$ + nhiễu $\sigma = 0.01$.

---

## 3. THUẬT TOÁN BƠM LỖI (ANOMALY INJECTION RECIPES)
Đầu vào là tín hiệu chuẩn $X_{norm}$, đầu ra là tín hiệu thực tế bị lỗi $X_{actual}$. Chúng ta sẽ bơm lỗi vào một khoảng thời gian $[t_{start}, t_{end}]$ ngẫu nhiên.

### A. DROPOUT (Mất tín hiệu / Tụt áp)
Mô phỏng sự cố đứt dây, nghẹt van hoàn toàn.
*   **Độ dài ($L$):** $10 \to 20$ steps.
*   **Thuật toán:**
    $$X_{actual}[t_{start}:t_{end}] = 0 + \mathcal{N}(0, \sigma_{floor})$$
    *(Ghi đè hoàn toàn vùng lỗi bằng nhiễu nền cực nhỏ, xấp xỉ 0).*

### B. SURGE (Gai vọt / Quá tải)
Mô phỏng sự cố phóng điện, kẹt cơ khí chớp nhoáng.
*   **Độ dài ($L$):** $1 \to 3$ steps (Rất ngắn để điểm `Impulse` vọt lên 1.0).
*   **Thuật toán:**
    $$X_{actual}[t_{start}:t_{end}] = X_{norm}[t_{start}:t_{end}] \times \text{Multiplier}$$
    *(Multiplier $\in [2.0, 3.0]$ đối với giá trị dương, hoặc $[0.1, 0.5]$ đối với giá trị âm như HV Voltage).*

### C. DEVIATION (Lệch dải vận hành)
Mô phỏng sự cố trôi cảm biến, sai lệch áp suất điều áp.
*   **Độ dài ($L$):** $> 40$ steps (Kéo dài để điểm `StaticBias` đạt tối đa).
*   **Thuật toán:**
    $$X_{actual}[t_{start}:t_{end}] = X_{norm}[t_{start}:t_{end}] + \text{Offset}$$
    *(Cộng thêm một hằng số $Offset$ đủ lớn để vượt qua `min_error_threshold` trong Config).*

### D. INSTABILITY (Chập chờn / Răng cưa)
Mô phỏng sự cố lỏng giắc cắm, hỏng vòng bi, nhiễu điện từ (EMI).
*   **Độ dài ($L$):** $> 20$ steps.
*   **Thuật toán:**
    $$X_{actual}[t_{start}:t_{end}] = X_{norm}[t_{start}:t_{end}] + \mathcal{N}(0, \sigma_{high}) + A \sin(2\pi f t)$$
    *(Cộng thêm nhiễu Gaussian có Variance gấp 5-10 lần bình thường, kết hợp sóng Sin tần số cao để đánh lừa thuật toán `Total Variation`).*

### E. TRANSITION (Phản hồi trễ / Ì máy)
Mô phỏng sự cố van khí nén mở chậm, ma sát cơ khí tăng.
*   **Độ dài ($L$):** Áp dụng tại các điểm giao cắt (Edges) của tín hiệu.
*   **Thuật toán 1 - Trễ pha (Phase Shift):**
    $$X_{actual}[t_{start}:t_{end}] = X_{norm}[t_{start} - \delta : t_{end} - \delta]$$
    *(Dịch mảng sang phải $\delta$ steps, với $\delta \in [3, 8]$ để điểm `TemporalLag` bắt được).*
*   **Thuật toán 2 - Ì đạo hàm (Sluggishness):**
    Áp dụng Moving Average Filter với window lớn (ví dụ: 10) lên $X_{norm}$ tại vùng có độ dốc cao $\to$ Làm bẹp độ dốc $\to$ Điểm `GradientInertia` sẽ vọt lên.

---

## 4. QUY TRÌNH KẾT HỢP (PHASE-AWARE INJECTION)
Để sinh ra đúng 15 nhãn lỗi, script Generator của bạn sẽ chọn ngẫu nhiên:
1.  **Chọn 1 Phase:** Ví dụ `STARTUP`. Ta có $t_{start} \in [5, 20]$.
2.  **Chọn 1 Pattern:** Ví dụ `SURGE`.
3.  **Chọn 1 Sensor:** Ví dụ `high_pressure_voltage`.
4.  **Thực thi:** Gọi hàm `inject_surge(X_norm, t_start)` lên cảm biến đó.
5.  **Lưu kết quả:** Lưu $X_{norm}$ dưới dạng `recon_raw` (giả lập AI Baseline) và $X_{actual}$ (sau khi bơm lỗi) dưới dạng `actual_raw`.