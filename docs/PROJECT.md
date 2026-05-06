# TÀI LIỆU TỔNG QUAN DỰ ÁN: HỆ SINH THÁI PHÁP Y ROBOT TỰ HÀNH
**(Autonomous Robot Forensics Ecosystem - V2026)**

## 1. BỐI CẢNH VÀ BÀI TOÁN (CONTEXT & PROBLEM STATEMENT)
*   **Môi trường vận hành:** Xưởng sơn ô tô (Automotive Paint Shop) với 47 Robot công nghiệp hoạt động song song.
*   **Đặc thù Dữ liệu:** 
    *   **OT Data (Vật lý):** Dữ liệu chuỗi thời gian đa biến (Multivariate Time-series) từ 6 cảm biến cốt lõi (Tốc độ motor, Khí nén, Dòng điện, Điện áp, Áp suất, Lưu lượng sơn). Tần suất thu thập cao 5Hz (0.2s/mẫu). Mỗi chu kỳ sơn một xe kéo dài khoảng 48 giây (~240 steps).
    *   **IT Data (Văn bản):** Lịch sản xuất (MES/ERP) và Kho tàng tri thức bảo trì quá khứ dưới dạng văn bản không cấu trúc (PM Cards).
*   **Bài toán đặt ra:** Hệ thống bảo trì truyền thống dựa trên "Ngưỡng cứng" (Hard Thresholds) đã hoàn toàn thất bại trước sự đa dạng của các chương trình sơn. Nhà máy cần một hệ thống AI có khả năng:
    1.  **Phát hiện sớm (Early Detection):** Bắt được các lỗi vi mô (Micro-anomalies) chỉ kéo dài 0.2s.
    2.  **Tiên lượng (Proactive Foresight):** Nhận diện sự suy thoái từ từ (Drift) trước khi máy hỏng.
    3.  **Pháp y tự động (Autonomous Forensics):** Không chỉ báo "Có lỗi", mà phải giải thích được "Lỗi gì, nguyên nhân từ đâu, và xử lý thế nào" bằng ngôn ngữ của con người.

## 2. THÁCH THỨC KỸ THUẬT CỐT LÕI (CORE CHALLENGES)
1.  **Pha loãng tín hiệu (Signal Dilution):** Một xung lỗi 0.2s nằm trong chu kỳ 48s rất dễ bị các mô hình AI thông thường "hòa tan" và bỏ qua.
2.  **Khuếch đại nhiễu (False Positives on Flatlines):** Các phương pháp chuẩn hóa cục bộ thường biến một nhiễu điện từ cực nhỏ trên một đường sóng phẳng thành một "gai lỗi" khổng lồ, gây bội thực cảnh báo (Alert Fatigue).
3.  **Dữ liệu không nhãn & Lẫn tạp chất (Contaminated Unlabeled Data):** 98% dữ liệu là bình thường, nhưng vẫn lẫn 2% dữ liệu lỗi không được dán nhãn. AI phải tự học cách phân biệt mà không bị "học vẹt" các lỗi có sẵn.
4.  **Khoảng cách Ngữ nghĩa (The Semantic Gap):** OT Data là những con số toán học vô tri. IT Data (PM Cards) là ngôn ngữ tự nhiên của thợ máy. Làm sao để AI dùng sóng vật lý để tra cứu tài liệu văn bản?

## 3. TẦM NHÌN KIẾN TRÚC TỔNG THỂ (MACRO ARCHITECTURE VISION)
Dự án áp dụng triết lý **Decoupled Intelligence (Trí tuệ tách biệt)**, chia hệ thống thành 2 thực thể chuyên biệt để tối ưu hóa hiệu suất và độ an toàn:
*   **Thực thể Cảm nhận (Perception Layer):** Đảm nhiệm bởi mô hình Deep Learning **GLC-Flow**. Làm việc trực tiếp với dữ liệu OT 5Hz để tìm ra các vi phạm vật lý với độ chính xác thống kê tuyệt đối.
*   **Thực thể Nhận thức (Cognition Layer):** Đảm nhiệm bởi **SignalHub** và **LLM Agent**. Làm việc với dữ liệu IT, đánh giá rủi ro kinh doanh và đưa ra quyết định bảo trì.

---

## 4. TRỤ CỘT 1: MÔ HÌNH DEEP LEARNING "GLC-FLOW" V7 (TẦNG CẢM NHẬN)
*(Global-Local Contextual Flow Matching Anomaly Detector)*

Vượt qua giới hạn của các mô hình Autoencoder dựa trên tái tạo lỗi (như VQ-Mixer V6), dự án áp dụng công nghệ Generative AI tiên tiến: **Continuous Normalizing Flow (CNF)**.

*   **Đơn vị quan sát:** Nhìn toàn cảnh 1 file (240 steps), không cắt vụn mù quáng.
*   **Dual-Stream Encoder (Bộ mã hóa luồng kép):**
    *   *Global Stream:* Chuẩn hóa trên toàn file (Global RevIN) $\rightarrow$ Bắt lỗi trôi (Drift) và triệt tiêu báo động giả trên đường bằng.
    *   *Local Stream:* Chuẩn hóa trên từng Patch (Local RevIN) $\rightarrow$ Bắt lỗi gai nhọn cục bộ (Spike).
*   **Exact Log-Likelihood (Độ tin cậy tuyệt đối):** Thay vì dùng "Reconstruction Error" (một con số vô nghĩa về mặt thống kê), GLC-Flow tính toán ra **Xác suất mật độ (p-value)**. Hệ thống có thể tự tin tuyên bố: *"File này có 99.7% xác suất là bất thường"*.
*   **Robust Thresholding:** Áp dụng phương pháp MAD (Median Absolute Deviation) để tự động tìm ngưỡng cảnh báo, miễn nhiễm với dữ liệu nhiễu trong tập huấn luyện.
*   **Exact Localization:** Dùng kỹ thuật *Log-Likelihood Perturbation* (Che giấu từng vùng để thử) nhằm khoanh vùng chính xác 100% cảm biến và thời điểm gây lỗi, thay thế cho các phương pháp Attention Heatmap thiếu tin cậy.

---

## 5. TRỤ CỘT 2: HỆ THỐNG MULTI-AGENT VÀ ĐIỀU PHỐI (TẦNG SUY LUẬN)
*(Autonomous Industrial Orchestration)*

Dữ liệu từ GLC-Flow không được gửi trực tiếp cho con người, mà đi qua một hệ sinh thái Đại lý tự hành chạy trên nền tảng Simulation-first (FastAPI, SQLite, DSPy).

### 5.1. Tuyến phân loại (TriageAgent) & Engine Điều phối
*   **SimulationRunner & Engine:** Điều phối luồng thời gian thực tế ảo, bắt sự kiện bùng nổ theo cụm (Burst L1) và xu hướng leo thang (Slope L2) kết hợp cơ giới chặn độ trễ (Debounce) để giảm nhiễu cảnh báo.
*   **TriageAgent:** Lớp chặn đầu tay bằng Rule-based thuần túy (không mất chi phí LLM). Quản lý định tuyến nhanh chóng các sự kiện: Tự động xếp xó (Auto-resolve) hoặc tạo yêu cầu chẩn đoán (Queue for Diagnostic).
*   *Mục tiêu:* Tránh quá tải Agent cốt lõi, chỉ thức tỉnh AI với rủi ro có tính xác thực cao.

### 5.2. Lõi pháp y (DiagnosticAgent)
Tuy LLM Agent chính không dùng ReAct thông thường, mà chạy framework **DSPy (Chain of Thought)** với tiến trình ràng buộc:
*   **Planning-First Workflow:** Agent ép buộc phải lập kế hoạch (Plan) hành động trước, không được phép gọi Tool tùy tiện, nhằm đảm bảo khả năng Audit và theo dõi tiến độ trên UI.
*   **Progressive Skill Loading:** Chỉ nạp ngữ cảnh hướng dẫn (SKILL.md) theo từng bước thay vì đẩy cùng lúc, nhằm loại bỏ hiện tượng ảo giác (Hallucination) của LLM giữa các bài toán khác nhau.
*   **Semantic Bridging (RAG cục bộ):** Truy vấn kho dự liệu PM Cards đã vector hóa bằng ChromaDB để tra cứu thông tin chuyên môn.
*   **Cổng kiểm duyệt Thread-safe (HITL Gate):** Các công cụ được phân cấp rủi ro (Low, High, Critical, Blocked). Gọi Tool nguy hiểm sẽ khóa luồng AI (Thread blocked) đến khi con người phê duyệt qua Web UI.

### 5.3. Trạm giám sát tổng (SystemicAgent)
*   Thực thể phụ trách xử lý khủng hoảng dây chuyền (Alert storms, vd: 5 thiết bị hỏng cùng lúc). Hiện đang ở giai đoạn chuẩn hóa thiết kế interface.

---

## 6. QUY TRÌNH VẬN HÀNH TIÊU CHUẨN (END-TO-END WORKFLOW)
1.  **OT Data Flow:** Robot R-07 hoàn thành 1 chu kỳ sơn (48s). Dữ liệu đẩy vào mô hình GLC-Flow.
2.  **Detection:** GLC-Flow phát hiện một gai áp suất bất thường. Log-likelihood giảm mạnh, xác suất lỗi là 99.9%. Tín hiệu đẩy vào Event Bus.
3.  **SignalHub Convergence:** SignalHub nhận thấy lỗi này lặp lại 5 lần trong 1 giờ qua (Urgency cao) và xe Maybach sắp vào chuyền (Impact cao). Tính điểm U-S-I vượt ngưỡng $\rightarrow$ Kích hoạt Agent.
4.  **Agentic Forensics:** 
    *   Agent trích xuất tọa độ lỗi.
    *   Dùng HyDE RAG tra cứu PM Cards $\rightarrow$ Chẩn đoán: "Nghẹt béc phun do mạt sơn".
5.  **Human Resolution:** Quản đốc nhận cảnh báo trên MS Teams kèm hình ảnh khoanh vùng lỗi và nguyên nhân. Bấm "Approve" để kỹ sư bảo trì ra tay trước khi xe Maybach tiến vào buồng sơn.

---
Dù ở MVP mô phỏng sử dụng cấu trúc gọn nhẹ (FastAPI, SQLite, ChromaDB), hệ thống tuân thủ kiến trúc tách biệt cho phép mở rộng lên nền tảng Native Databricks (Delta Tables, Unity Catalog, Model Serving) cho quy mô hàng trăm Robot phần cứng sau này.
*   **Tương lai (2026+):** Hoàn thiện SystemicAgent xử lý rủi ro diện rộng và hòa mạng Agentic Web (Moltbook), cho phép các Agent chủ động giao tiếp
*   **Zero False Positives:** Loại bỏ hoàn toàn sự phiền toái của các cảnh báo giả, khôi phục niềm tin của đội ngũ vận hành vào hệ thống AI.
*   **Predictive to Prescriptive:** Không chỉ dừng ở việc "Dự đoán khi nào hỏng", hệ thống tự động "Đề xuất cách sửa chữa" dựa trên tri thức lịch sử của chính nhà máy.
*   **Khả năng mở rộng (Scalability):** Kiến trúc Native Databricks (Delta Tables, Unity Catalog, Model Serving) cho phép dễ dàng scale lên hàng trăm Robot hoặc mang sang các nhà máy khác mà không cần đập đi xây lại hạ tầng.
*   **Tương lai (2026+):** Sẵn sàng tích hợp vào mạng lưới Agentic Web (Moltbook), nơi các Agent nhà máy có thể giao tiếp, trao đổi kỹ năng (Skills) và tự động đàm phán mua sắm vật tư thay thế.