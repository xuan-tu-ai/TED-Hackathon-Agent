# TÀI LIỆU TỔNG QUAN DỰ ÁN: HỆ SINH THÁI PHÁP Y ROBOT HÀN TỰ HÀNH
**(Autonomous Welding Robot Forensics Ecosystem - V2026)**

## 1. BỐI CẢNH VÀ BÀI TOÁN (CONTEXT & PROBLEM STATEMENT)
*   **Môi trường vận hành:** Dây chuyền hàn công nghiệp với 4 Robot hàn (R-00 đến R-03) hoạt động liên tục theo ca.
*   **Đặc thù Dữ liệu:**
    *   **OT Data (Vật lý):** Dữ liệu chuỗi thời gian đa biến (Multivariate Time-series) từ 3 cảm biến cốt lõi của mỗi robot hàn:
        *   `wire_feed_speed` — Tốc độ cấp dây hàn (m/min)
        *   `wire_temperature` — Nhiệt độ dây hàn (°C)
        *   `welding_current` — Dòng điện hàn (A)
    *   **IT Data (Văn bản):** Kho tàng tri thức bảo trì quá khứ dưới dạng PM Cards (Phiếu bảo trì) lưu trên Databricks Delta Lake.
*   **Bài toán đặt ra:** Hệ thống giám sát truyền thống dựa trên "Ngưỡng cứng" (Hard Thresholds) thất bại trước sự biến thiên phức tạp của tín hiệu hàn. Nhà máy cần một hệ thống AI có khả năng:
    1.  **Phát hiện sớm (Early Detection):** Bắt được 5 loại bất thường đặc trưng của hàn công nghiệp: `spike`, `cable_wear`, `flatline`, `thermal_drift`, `highfreq`.
    2.  **Tiên lượng (Proactive Foresight):** Nhận diện sự suy thoái từ từ (cable_wear, thermal_drift) trước khi robot dừng máy đột ngột.
    3.  **Pháp y tự động (Autonomous Forensics):** Không chỉ báo "Có lỗi", mà phải giải thích được "Lỗi gì, cảm biến nào, nguyên nhân từ đâu, và xử lý thế nào" bằng ngôn ngữ của kỹ sư bảo trì.

## 2. THÁCH THỨC KỸ THUẬT CỐT LÕI (CORE CHALLENGES)
1.  **Pha loãng tín hiệu (Signal Dilution):** Một xung gai (spike) tốc độ dây cực ngắn trong chuỗi hàn dài rất dễ bị các mô hình AI thông thường "hòa tan" và bỏ qua.
2.  **Khuếch đại nhiễu (False Positives on Flatlines):** Chuẩn hóa cục bộ biến tín hiệu đường phẳng (flatline — mất tín hiệu dây hàn) thành "gai lỗi" khổng lồ, gây bội thực cảnh báo (Alert Fatigue).
3.  **Dữ liệu không nhãn & Lẫn tạp chất (Contaminated Unlabeled Data):** Phần lớn dữ liệu là bình thường, nhưng lẫn các bất thường chưa được dán nhãn. AI phải tự học cách phân biệt mà không bị "học vẹt" các lỗi có sẵn.
4.  **Khoảng cách Ngữ nghĩa (The Semantic Gap):** OT Data là các con số vật lý (A, °C, m/min). IT Data (PM Cards) là ngôn ngữ tự nhiên của thợ hàn. Làm sao để AI dùng tín hiệu vật lý để tra cứu tài liệu văn bản và đưa ra chẩn đoán có nghĩa?

## 3. TẦM NHÌN KIẾN TRÚC TỔNG THỂ (MACRO ARCHITECTURE VISION)
Dự án áp dụng triết lý **Decoupled Intelligence (Trí tuệ tách biệt)**, chia hệ thống thành 2 thực thể chuyên biệt:
*   **Thực thể Cảm nhận (Perception Layer):** Đảm nhiệm bởi mô hình Deep Learning **GLCMaeV9**. Làm việc trực tiếp với dữ liệu OT 3 cảm biến để phát hiện và định lượng bất thường với điểm p99 theo từng cảm biến.
*   **Thực thể Nhận thức (Cognition Layer):** Đảm nhiệm bởi **WeldBot AI** (SimulationEngine + DSPy Agent). Làm việc với dữ liệu IT, đánh giá rủi ro vận hành và đưa ra quyết định bảo trì có thể kiểm toán.

---

## 4. TRỤ CỘT 1: MÔ HÌNH DEEP LEARNING "GLCMaeV9" (TẦNG CẢM NHẬN)
*(Global-Local Contextual Masked Autoencoder — V9)*

Vượt qua giới hạn của các mô hình Autoencoder chuẩn hóa cục bộ (RevIN) dễ bị nhiễu trên tín hiệu phẳng, GLCMaeV9 áp dụng chiến lược chuẩn hóa toàn cục:

*   **LogScaler + GlobalSeqNorm:** Chuẩn hóa log-scale trên toàn bộ sequence thay vì per-patch, giúp tín hiệu `flatline` không bị khuếch đại thành lỗi giả.
*   **Chiến lược 4-dataset:** Train → Val → Threshold Calibration → Test, tách biệt hoàn toàn dữ liệu hiệu chỉnh ngưỡng khỏi dữ liệu huấn luyện.
*   **p99 Aggregation (Per-sensor scoring):** Điểm bất thường được tổng hợp ở phân vị thứ 99 cho từng cảm biến độc lập:
    ```json
    {
      "p99_score_wire_feed_speed": 0.88,
      "p99_score_wire_temperature": 0.05,
      "p99_score_welding_current": 0.04
    }
    ```
    Cảm biến có điểm cao nhất là cảm biến **dominant** — xác định loại bất thường và họ chẩn đoán.
*   **Threshold Calibration:** `threshold_config_v9.json` lưu ngưỡng p99 riêng cho từng cảm biến, được hiệu chỉnh trên tập dữ liệu calibration độc lập.
*   **5 loại bất thường được phát hiện:**

    | Loại | Cảm biến dominant | Đặc trưng |
    |---|---|---|
    | `spike` | `wire_feed_speed` | Gai đột ngột, thời gian cực ngắn |
    | `cable_wear` | `wire_feed_speed` | Suy thoái từ từ, điểm leo thang dài hạn |
    | `flatline` | `wire_feed_speed` | Tín hiệu phẳng — mất liên lạc dây hàn |
    | `thermal_drift` | `wire_temperature` | Nhiệt độ trôi khỏi dải vận hành an toàn |
    | `highfreq` | `welding_current` | Dao động tần số cao trong dòng hàn |

---

## 5. TRỤ CỘT 2: HỆ THỐNG MULTI-AGENT "WELDBOT AI" (TẦNG SUY LUẬN)
*(Autonomous Industrial Orchestration)*

Điểm p99 từ GLCMaeV9 không được gửi trực tiếp cho con người, mà đi qua một hệ sinh thái Đại lý tự hành chạy trên nền tảng Databricks + SQLite + DSPy.

### 5.1. Tuyến phân loại (TriageAgent) & Engine Điều phối
*   **SimulationRunner & Engine:** Điều phối luồng thời gian thực tế ảo, đọc `PerceptionResult` từ Databricks Delta Lake, bắt sự kiện bùng nổ theo cụm (Burst L1) và xu hướng leo thang (Slope L2) kết hợp cơ chế chặn độ trễ (Debounce) để giảm nhiễu cảnh báo.
*   **V9 Score Routing:** `_create_incident` đọc `per_sensor_scores` từ sự kiện L1 gần nhất, gọi `dominant_sensor()` để xác định cảm biến thủ phạm và `sensor_to_family()` để ánh xạ sang họ chẩn đoán (`DROPOUT`, `DEVIATION`, `INSTABILITY`).
*   **TriageAgent:** Lớp chặn đầu tay bằng Rule-based thuần túy (không mất chi phí LLM). Định tuyến nhanh: Tự động xếp xó (Auto-resolve) hoặc tạo yêu cầu chẩn đoán (Queue for Diagnostic).
*   *Mục tiêu:* Tránh quá tải Agent cốt lõi, chỉ thức tỉnh AI với rủi ro có tính xác thực cao.

### 5.2. Lõi pháp y (DiagnosticAgent)
LLM Agent chính chạy framework **DSPy (Chain of Thought)** với tiến trình ràng buộc:
*   **Planning-First Workflow:** Agent bắt buộc lập kế hoạch (Plan) hành động trước, không được gọi Tool tùy tiện, đảm bảo khả năng Audit và theo dõi tiến độ trên UI.
*   **V9 Score Context:** Agent nhận điểm p99 từng cảm biến trong context chẩn đoán, cho phép suy luận trực tiếp từ tín hiệu vật lý thay vì chỉ dựa vào nhãn severity.
*   **Progressive Skill Loading:** Chỉ nạp SKILL.md hướng dẫn chẩn đoán đúng loại bất thường (`spike.md`, `cable_wear.md`, `flatline.md`, `thermal_drift.md`, `highfreq.md`) theo từng bước.
*   **Semantic Bridging (RAG):** Truy vấn PM Cards từ Databricks (ILIKE SQL) với ChromaDB làm fallback, tra cứu thông tin bảo trì chuyên biệt theo cảm biến và loại lỗi.
*   **Cổng kiểm duyệt Thread-safe (HITL Gate):** Các công cụ được phân cấp rủi ro (Low, High, Critical, Blocked). Gọi Tool nguy hiểm (vd: `stop_welding_cycle`) sẽ khóa luồng AI đến khi kỹ sư phê duyệt qua Web UI.

### 5.3. Trạm giám sát tổng (SystemicAgent)
*   Thực thể phụ trách xử lý khủng hoảng dây chuyền (Alert storms, vd: nhiều robot hàn báo lỗi cùng lúc). Hiện đang ở giai đoạn chuẩn hóa thiết kế interface.

---

## 6. QUY TRÌNH VẬN HÀNH TIÊU CHUẨN (END-TO-END WORKFLOW)
1.  **OT Data Flow:** Robot R-00 hoàn thành 1 chuỗi hàn. GLCMaeV9 chạy inference, ghi kết quả `per_sensor_scores` vào bảng `perception_results` trên Databricks Delta Lake.
2.  **Detection:** Điểm `p99_score_wire_feed_speed = 0.88` vượt ngưỡng calibration. L1 Poller phát hiện burst — 3 sự kiện bất thường trong cửa sổ 1 giờ.
3.  **Incident Creation:** `_create_incident` đọc `per_sensor_scores` từ L1 event gần nhất, xác định dominant sensor là `wire_feed_speed` (score 0.88), ánh xạ sang họ `DROPOUT`, tạo `DiagnosticIncident` với severity `CRITICAL`.
4.  **Agentic Forensics:**
    *   DiagnosticAgent nhận context chẩn đoán bao gồm điểm p99 từng cảm biến.
    *   Nạp `cable_wear.md` skill để hướng dẫn chẩn đoán suy thoái dây hàn.
    *   Truy vấn PM Cards trên Databricks → Tìm thấy thẻ bảo trì liên quan đến mòn cáp cấp dây robot R-00.
    *   Chẩn đoán: *"Cable wear on wire feed mechanism — gradual degradation pattern detected. Recommend inspection of drive rollers and cable liner."*
5.  **Human Resolution:** Kỹ sư nhận cảnh báo trên Web UI kèm điểm p99 từng cảm biến, chẩn đoán nguyên nhân và đề xuất hành động. Bấm "Approve" để ghi nhận lệnh bảo trì trước khi robot hàn bị dừng máy không kiểm soát.

---

Dù ở MVP mô phỏng sử dụng cấu trúc gọn nhẹ (FastAPI, SQLite, ChromaDB), hệ thống tuân thủ kiến trúc tách biệt cho phép mở rộng lên nền tảng Native Databricks (Delta Tables, Unity Catalog, Model Serving) cho quy mô hàng chục đến hàng trăm robot hàn sau này.
*   **Zero False Positives:** LogScaler + GlobalSeqNorm loại bỏ báo động giả trên tín hiệu flatline, khôi phục niềm tin của đội ngũ vận hành vào hệ thống AI.
*   **Predictive to Prescriptive:** Không chỉ dừng ở "Dự đoán khi nào hỏng", hệ thống tự động "Đề xuất cách sửa chữa" dựa trên tri thức PM Cards của chính nhà máy.
*   **Khả năng mở rộng (Scalability):** Kiến trúc Databricks-first (Delta Tables, Unity Catalog) cho phép scale lên hàng trăm robot hoặc mang sang các dây chuyền hàn khác mà không cần đập đi xây lại hạ tầng.
*   **Tương lai (2026+):** Sẵn sàng tích hợp vào mạng lưới Agentic Web (Moltbook), nơi các Agent nhà máy có thể giao tiếp, trao đổi kỹ năng (Skills) và tự động đặt lịch bảo trì phòng ngừa.
