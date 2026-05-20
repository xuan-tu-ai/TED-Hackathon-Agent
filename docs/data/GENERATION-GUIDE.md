# YÊU CẦU KỸ THUẬT: SCRIPT SINH DỮ LIỆU MÔ PHỎNG (WELDING ROBOT)
**Dự án:** GLC-Anomaly-MaskFormer Hackathon
**Mục tiêu:** Sinh tập dữ liệu Time-series đa biến mô phỏng hoạt động của Robot Hàn hồ quang/Hàn điểm. Dữ liệu phải phản ánh đúng quán tính nhiệt, độ trễ dòng điện và các kịch bản lỗi hàn đặc thù.

## 1. QUY CÁCH DỮ LIỆU CHUNG (DATA SHAPE & METADATA)
*   **Kích thước 1 Sample:** `[3, 240]` (3 Cảm biến, 240 steps). Đây là kích thước chuẩn, nhưng robot bản thân có thể biến động, nên có thể thay đổi lên 300, 400, hoặc giảm xuống 200.
*   **Metadata đi kèm:** 
    *   `robot_id`: Random Int từ `[0, 3]` (4 robots).
    *   `program_id`: Random Int từ `[0, 15]` (Đại diện cho các chu trình hàn dài/ngắn khác nhau).
*   **Danh sách 3 Cảm biến (Sensors):**
    *   *Nhóm Nguyên nhân (Actuator):* `S0: wire_feed_speed` (Tốc độ cấp dây / Tốc độ quay).
    *   *Nhóm Kết quả (Responses):* `S1: welding_current` (Dòng điện hàn), `S2: temperature` (Nhiệt độ mối hàn/đầu mỏ hàn).

## 2. QUY LUẬT VẬT LÝ & ĐỘNG HỌC (PHYSICS & DYNAMICS)
Dữ liệu sinh ra không được là các đường thẳng đơn giản, mà phải tuân thủ hệ thống LTI (Linear Time-Invariant):
1.  **Động học Dòng điện (Fast Response & Overshoot):**
    *   `welding_current` bám theo `wire_feed_speed` với độ trễ rất nhỏ (1-2 steps).
    *   Khi tốc độ dây vọt lên (bắt đầu hàn), dòng điện phải có **Overshoot** (vọt lố khoảng 10-15% để mồi hồ quang - Arc ignition) rồi dao động tắt dần cực nhanh về mức ổn định.
2.  **Động học Nhiệt độ (Slow Response & High Inertia):**
    *   `temperature` phụ thuộc vào `welding_current` nhưng có **quán tính nhiệt rất lớn**.
    *   Khi dòng điện tăng, nhiệt độ không tăng thẳng đứng mà tăng theo **đường cong hàm mũ (Exponential charge curve)** giống như tụ điện sạc. Khi ngừng hàn, nhiệt độ giảm từ từ (Cooling curve).
3.  **Hệ quả của Metadata:**
    *   `robot_id` tạo ra sai số Baseline: Ví dụ, hệ thống làm mát của Robot 3 kém hơn nên `temperature` luôn cao hơn Robot 0 khoảng 5 độ.

## 3. ĐẶC TẢ TẬP HUẤN LUYỆN (TRAIN SET - 100% NORMAL)
Tập Train chỉ chứa dữ liệu Bình thường, bao gồm:
*   **Nhiễu hỗn hợp (Colored Noise):** 
    *   `wire_feed_speed` có nhiễu rung động cơ khí (Sinusoidal tần số cao biên độ nhỏ).
    *   `welding_current` có nhiễu trắng (White noise) mạnh do hồ quang hàn không ổn định.
    *   `temperature` có nhiễu trôi dạt (Random walk) rất mượt, hiếm khi có nhiễu gai.
*   **Sự kiện vi mô bình thường (Normal Micro-events):** Tại thời điểm bắt đầu hàn (ví dụ step 30), `welding_current` có một dao động giật cục (Arc Strike Fluctuation) kéo dài 3 steps. Mô hình phải học được đây là hiện tượng "Mồi hồ quang" bình thường.

## 4. ĐẶC TẢ TẬP KIỂM THỬ (TEST SET - NORMAL + ABNORMAL)
Tập Test gồm 50% file Normal và 50% file Abnormal. Các kịch bản lỗi (Anomalies) phải bám sát thực tế ngành hàn:

*   **Lỗi 1: Dính dây hàn / Ngắn mạch (Cross-sensor Spike):** 
    *   *Mô tả:* Tiêm một Spike cực lớn (kéo dài 2 steps) vào `welding_current` ở giữa chu kỳ hàn. 
    *   *Điểm mấu chốt:* `wire_feed_speed` VẪN BÌNH THƯỜNG. Đây là lỗi dây hàn chạm vật hàn gây đoản mạch.
*   **Lỗi 2: Tích tụ nhiệt bất thường (Thermal Runaway / Drift):** 
    *   *Mô tả:* Trong nửa sau của chu kỳ, `temperature` không đi ngang mà tiếp tục tăng tuyến tính vuợt mức bình thường.
    *   *Điểm mấu chốt:* `welding_current` và `wire_feed_speed` hoàn toàn bình thường. Lỗi này do hỏng hệ thống nước làm mát mỏ hàn. (Test khả năng bắt Drift của luồng Global RevIN).
*   **Lỗi 3: Mất phản hồi cảm biến (Sensor Flatline):** 
    *   *Mô tả:* `wire_feed_speed` đang ở mức 4000, đột nhiên tụt về 0 và đi thẳng tắp trong 20 steps, sau đó phục hồi. Dòng điện và nhiệt độ vẫn tương đối ổn định do quán tính.
    *   *Điểm mấu chốt:* Test khả năng Local RevIN khuếch đại sự cố đứt tín hiệu.
*   **Lỗi 4: Lệch pha nguồn hàn (Phase Shift Anomaly):** 
    *   *Mô tả:* `wire_feed_speed` đã tăng, nhưng `welding_current` bị trễ tới 15 steps mới tăng theo (thay vì 1-2 steps như bình thường). 
    *   *Điểm mấu chốt:* Hình dáng sóng dòng điện không đổi, chỉ bị đẩy lùi về sau. (Test khả năng học Temporal Attention của Transformer để bắt lỗi độ trễ cấp nguồn).

## 5. ĐỊNH DẠNG ĐẦU RA (OUTPUT FORMAT)
File csv giống [DATA](../../data/mock_welding_robot_data.csv)

***

### 💡 Lời khuyên cho Data Engineer:
Với 3 cảm biến này, mô hình **GLC-Anomaly-MaskFormer** của bạn sẽ "tỏa sáng rực rỡ" nhất ở **Lỗi số 1 và Lỗi số 2**. 
*   Việc che (mask) cảm biến Dòng điện, và bắt Transformer phải dùng Tốc độ cấp dây để vẽ lại cái vọt lố (Overshoot) lúc mồi hồ quang sẽ chứng minh sức mạnh học Nhân quả (Causal Learning) tuyệt đối của kiến trúc.

Bạn đưa Prompt này cho LLM để sinh code Python dùng `scipy.signal.TransferFunction` hoặc các hàm ODE cơ bản, bạn sẽ có ngay một tập Mock Data hoàn hảo cho Hackathon!