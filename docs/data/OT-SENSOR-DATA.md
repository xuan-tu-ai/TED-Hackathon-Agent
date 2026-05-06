# ĐẶC TẢ DỮ LIỆU THÔ (RAW OT DATA) - HACKATHON EDITION

## 1. Cấu trúc Bảng Dữ liệu Chuỗi thời gian (Continuous Time-Series Table)
Trước khi cắt thành các window 32-steps, dữ liệu gốc được lưu dưới dạng chuỗi thời gian liên tục. Một file/chu kỳ sơn (Carbody) kéo dài **48 giây**, tần suất **5Hz (0.2s/sample)** $\to$ Mỗi chu kỳ có **240 dòng**.

| Tên Cột (Column) | Kiểu dữ liệu (Type) | Ví dụ (Example) | Mô tả (Description) |
| :--- | :--- | :--- | :--- |
| `robot_code` | String | `"R-05"` | Mã định danh Robot (R-01 đến R-47). |
| `program_number` | Integer | `42` | Mã chương trình sơn (Đại diện cho loại xe/màu sơn). |
| `carbody_id` | String | `"CAR-2025-X9"` | ID duy nhất của một chu kỳ sơn (1 xe). |
| `timestamp` | Timestamp | `2026-05-06 10:00:00.000` | Tăng đều 0.2 giây mỗi dòng. |
| `step_index` | Integer | `0` đến `239` | Vị trí tương đối trong chu kỳ. |
| `air_motor_rotation_speed`| Float | `250.5` | Cảm biến 1 (Thứ tự phải cố định). |
| `electro_pneumatic_output`| Float | `450.2` | Cảm biến 2 |
| `high_pressure_current` | Float | `85.4` | Cảm biến 3 |
| `high_pressure_voltage` | Float | `-90.0` | Cảm biến 4 |
| `paint_pressure` | Float | `0.15` | Cảm biến 5 |
| `paint_flow_rate` | Float | `250.0` | Cảm biến 6 |

---

## 2. Đặc tính Vật lý của 6 Cảm biến (Dùng để Mock Normal Data)
Để dữ liệu mock thuyết phục được giám khảo, các đường tín hiệu "Bình thường" (Normal) phải tuân theo các quy luật vật lý sau:

### Sensor 1: `air_motor_rotation_speed` (Tốc độ motor khí nén)
*   **Đơn vị:** RPM
*   **Dải giá trị:** `0` đến `300` (hoặc lên tới `3000` tùy program).
*   **Hành vi bình thường:** Có dạng **Bậc thang (Staircase)**. Khi thay đổi tốc độ, nó không nhảy vọt ngay lập tức mà có một độ dốc nhỏ (Ramp-up/Ramp-down). Ở trạng thái ổn định, nó có nhiễu dao động rất nhẹ ($\pm 2$ RPM).

### Sensor 2: `electro_pneumatic_output` (Đầu ra điều khiển khí nén)
*   **Đơn vị:** Internal Unit (0 - 4000)
*   **Dải giá trị:** `0` đến `4000`.
*   **Hành vi bình thường:** Tín hiệu này là lệnh điều khiển van nên nó **rất nhiễu và chập chờn (High-frequency ripple)**. Thường có một xung cực cao (khoảng 4000) ở những giây đầu tiên để xả van (purge), sau đó tụt xuống dao động mạnh quanh mức `300 - 800` trong suốt chu kỳ.

### Sensor 3: `high_pressure_current` (Dòng điện tĩnh điện)
*   **Đơn vị:** $\mu A$ (Micro-Ampere)
*   **Dải giá trị:** `0` đến `120`.
*   **Hành vi bình thường:** Bắt đầu từ 0. Khi Robot bắt đầu xịt sơn (giây thứ 5), nó tăng nhanh lên mức `80 - 100` và duy trì một đường cao nguyên (Plateau) khá mượt, nhiễu nền khoảng $\pm 5 \mu A$.

### Sensor 4: `high_pressure_voltage` (Điện áp tĩnh điện)
*   **Đơn vị:** $kV$ (Kilo-Volt)
*   **Dải giá trị:** `0` đến `-100` (Lưu ý: **Tín hiệu luôn ÂM**).
*   **Hành vi bình thường:** Dạng hình chữ U lộn ngược. Ở giây thứ 3, nó sụt thẳng từ `0` xuống `-90 kV`. Duy trì cực kỳ phẳng và ổn định (nhiễu $\pm 1 kV$) suốt quá trình sơn, sau đó vọt về `0` ở cuối chu kỳ.

### Sensor 5: `paint_pressure` (Áp suất sơn)
*   **Đơn vị:** MPa
*   **Dải giá trị:** `0.0` đến `0.3`.
*   **Hành vi bình thường:** Tương quan chặt chẽ với Sensor 2 và Sensor 6. Khi không phun, áp suất bằng 0. Khi phun, áp suất đạt khoảng `0.1` đến `0.2` MPa. Có các dao động nhỏ hình sin (Ripple) do hoạt động của bơm màng.

### Sensor 6: `paint_flow_rate` (Lưu lượng sơn)
*   **Đơn vị:** cc/min
*   **Dải giá trị:** `0` đến `400`.
*   **Hành vi bình thường:** Dạng **Xung vuông (Square waves)**. Robot không xịt sơn liên tục mà xịt theo từng nhát cắt (ví dụ: xịt 5 giây, nghỉ 2 giây, xịt 10 giây). Khi xịt, lưu lượng bật lên `250` hoặc `350` cc/min ngay lập tức.

---

## 3. Hướng dẫn sinh dữ liệu (Mocking Strategy) cho Hackathon

Để tạo ra một tập dataset hoàn chỉnh mang đi thi, bạn có thể viết 1 script Python với logic sau:

**Bước 1: Sinh dữ liệu "Chuẩn" (Golden Baseline)**
Tạo ra 1 hàm `generate_normal_cycle(program_id)`. Dùng hàm `np.piecewise` hoặc `scipy.signal.square` để tạo ra các đường bao (envelope) như đặc tả ở phần 2. Thêm nhiễu trắng `np.random.normal(0, noise_level)` vào để tín hiệu trông tự nhiên.

**Bước 2: Bơm "Lỗi" vào dữ liệu (Inject Anomalies)**
Tạo một vòng lặp sinh ra 1000 file (chu kỳ). Chọn ngẫu nhiên 87 file để bơm lỗi vào dựa trên 5 pattern:
*   `inject_dropout(signal)`: Ép 10 index ngẫu nhiên về 0.
*   `inject_surge(signal)`: X2 giá trị tại 2 index liên tiếp.
*   `inject_deviation(signal)`: Cộng hằng số 0.15 vào 50 index.
*   `inject_instability(signal)`: Cộng nhiễu `np.random.normal(0, high_variance)` vào 30 index.
*   `inject_transition(signal)`: `np.roll(signal, shift=5)` để tạo độ trễ pha.

**Bước 3: Chuyển đổi sang Bảng Gold (Slicing Windows)**
Để feed vào model VQ-VAE và Heuristic Engine của bạn, dữ liệu thô này cần được "cắt" thành các cửa sổ trượt:
*   Window Size ($W$): 32
*   Stride ($S$): 8
*   Từ 240 dòng của 1 file $\to$ Tạo ra khoảng 26 dòng (windows), mỗi dòng chứa một array phẳng `raw_values_array` kích thước $6 \times 32 = 192$ phần tử.