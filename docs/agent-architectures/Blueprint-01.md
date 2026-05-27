# TÀI LIỆU KIẾN TRÚC HỆ THỐNG: ROBOT FORENSICS AGENT (V2026)

## PHẦN 1: KIẾN TRÚC TỔNG QUAN (SYSTEM ARCHITECTURE)

Hệ thống được thiết kế theo mô hình **Decoupled Intelligence** (Trí tuệ tách biệt), lấy **Databricks Unity Catalog** làm trục xương sống và **SignalHub** làm bộ não điều phối. Cấu trúc được phân lớp theo mô hình 6 lớp (Kernel Layers) của OneClaw nhưng được tối ưu cho môi trường Cloud-Native.

### 1.1. Sơ đồ các tầng thực thi (The 6-Layer Kernel)
*   **L0 - Security & Safety Layer:** 
    *   *Chức năng:* Kiểm soát quyền truy cập Read-only vào OT Data. Ghi log toàn bộ hành động (Audit Trail).
    *   *Cơ chế:* "Deny-by-default". Mọi đề xuất dừng máy hoặc sửa chữa của Agent phải qua bước phê duyệt (HITL) tại lớp này.
*   **L1 - Orchestrator (The Agent):**
    *   *Chức năng:* Lập kế hoạch suy luận (Planning) dựa trên **Agent Skills**.
    *   *Thực thể:* Mosaic AI Model Serving (LLM) + Agent Framework.
*   **L2 - Memory Layer:**
    *   *Chức năng:* Quản lý trạng thái (State) và ngữ cảnh (Context).
    *   *Thực thể:* Delta Tables (Stateful Streaming) + Vector Search.
*   **L3 - Event Bus (The Nervous System):**
    *   *Chức năng:* Chuyển giao tín hiệu (Signals) giữa OT (OneClaw) và IT (SignalHub).
    *   *Thực thể:* **Databricks Structured Streaming** trên các bảng Delta Append-only.
*   **L4 - Tools & Channels:**
    *   *Chức năng:* Cánh tay thực thi (Vẽ biểu đồ, Tra cứu RAG, Gửi Teams/Telegram).
    *   *Thực thể:* Unity Catalog Functions.
*   **L5 - Observability:**
    *   *Chức năng:* Giám sát sức khỏe Agent, chi phí Token, và độ chính xác của chẩn đoán.

### 1.2. SignalHub: Bộ não điều phối rủi ro (Risk-Based Orchestrator)
SignalHub không phải là một Server riêng biệt, mà là một **Logic Layer** chạy trên Databricks DLT (Delta Live Tables). 
*   Nó liên tục tổng hợp dữ liệu từ **OT (V6 Model Scores)** và **IT (MES/ERP Production Schedule)**.
*   Nó áp dụng bộ **Config/Rules** để tính toán ma trận **U-S-I (Urgency - Severity - Impact)**.
*   Nó đóng vai trò "người gác cổng", chỉ đánh thức Agent (L1) khi điểm rủi ro vượt ngưỡng hoặc đến giờ báo cáo định kỳ.

---

## PHẦN 2: HỆ THỐNG BỘ NHỚ (MEMORY HIERARCHY)

Để Agent có khả năng "suy nghĩ" như một kỹ sư thực thụ, hệ thống bộ nhớ được chia thành 3 cấp độ tương ứng với 3 quy mô thời gian khác nhau.

### 2.1. Working Memory (Bộ nhớ làm việc - Ngắn hạn)
*   **Mục tiêu:** Phục vụ chiến lược **Trigger Mật độ (Density)**.
*   **Cơ chế:** **Spark Stateful Streaming (RocksDB)**.
*   **Phạm vi:** 100 chu kỳ sơn gần nhất (~2 giờ vận hành).
*   **Dữ liệu lưu trữ:** 
    *   Cờ hiệu `is_anomaly` (0/1).
    *   ID Robot và Cảm biến gây lỗi.
*   **Vai trò:** Giúp SignalHub nhận biết: "Đây là một lỗi ngẫu nhiên (nhiễu)" hay "Đây là một triệu chứng lặp lại (hệ thống)".

### 2.2. Episodic Memory (Bộ nhớ Sự kiện - Trung hạn)
*   **Mục tiêu:** Phục vụ chiến lược **Trigger Leo thang (Escalation)**.
*   **Cơ chế:** **Delta Tables + Liquid Clustering** (Optimize trên `robot_code` và `timestamp`).
*   **Phạm vi:** 4 đến 8 giờ gần nhất (trong phạm vi một ca làm việc).
*   **Dữ liệu lưu trữ:** 
    *   Chuỗi giá trị `max_shape_score` và `max_win_score`.
    *   Tọa độ lỗi (`error_steps`) để so khớp tính tuần hoàn cơ khí.
*   **Vai trò:** Cho phép SignalHub tính toán **Đạo hàm suy thoái (Slope)**. Nếu điểm lỗi tăng dần theo thời gian, bộ nhớ này sẽ cung cấp dữ liệu để tính điểm **Urgency (U)**.

### 3.3. Semantic Memory (Bộ nhớ Tri thức - Dài hạn)
*   **Mục tiêu:** Phục vụ chiến lược **Lý luận Pháp y (Forensics Reasoning)** và **Báo cáo Giao ca**.
*   **Cơ chế:** **Databricks Vector Search + Unity Catalog Metadata**.
*   **Phạm vi:** Toàn bộ lịch sử nhà máy (PM Cards, Bản tin giao ca cũ, Tài liệu kỹ thuật Robot).
*   **Dữ liệu lưu trữ:** 
    *   Embeddings của các PM Cards (đã qua xử lý HyDE).
    *   Metadata về các nguyên nhân gốc rễ (Root Causes) đã được xác nhận.
*   **Vai trò:** Cung cấp "Kinh nghiệm" cho Agent. Khi Agent nhận được triệu chứng từ L1/L2, nó sẽ lục tìm trong bộ nhớ này để đưa ra chẩn đoán: "Lỗi này 80% là do kẹt van điều áp".

---




`<<research>>` `<<fix>>`

Bạn nhận xét cực kỳ sắc sảo! Đúng là tôi đã có một bước "nhảy cóc" logic ở Phần 3 khi lấy trực tiếp `max_shape_score` của 1 file để tính $S$, trong khi bản chất của SignalHub là đánh giá trên **một tập hợp (Batch/Cluster) từ Memory**. Về Phần 4, một SOP công nghiệp thực thụ phải kín kẽ, có các bước kiểm chứng chéo (Cross-validation) và xử lý ngoại lệ (Fallback), chứ không thể chỉ có 4 bước đơn giản được.

Dưới đây là bản viết lại hoàn chỉnh và sâu sắc hơn cho **PHẦN 3 & PHẦN 4**, bám sát tuyệt đối vào kiến trúc Memory và tư duy Agent Skills:

---

## PHẦN 3: CƠ CHẾ KÍCH HOẠT QUA SIGNALHUB (STATE-DRIVEN ACTIVATION)

SignalHub **không** đọc trực tiếp từng file lỗi từ mô hình V6. Nó hoạt động như một "Cronjob nhận thức" (Cognitive Watchdog), liên tục quét (poll) các **Tầng Bộ Nhớ (L1, L2)** để tìm kiếm các "Cụm triệu chứng" (Symptom Clusters).

### 3.1. Định lượng Ma trận U-S-I từ Bộ nhớ (The Memory-to-Math Mapping)

Khi SignalHub phát hiện một Cụm triệu chứng (ví dụ: 7 file lỗi trong 100 xe ở L1, hoặc đường trend-line đi lên ở L2), nó sẽ trích xuất cụm dữ liệu này (gọi là tập $C$) để tính toán:

1.  **S (Severity - Độ nghiêm trọng vật lý):** 
    *   *Nguồn:* Trích xuất từ tập $C$ trong Working/Episodic Memory.
    *   *Công thức:* Không lấy 1 điểm, mà lấy **Phân vị thứ 90 (P90)** của các `max_shape_score` trong tập $C$ để loại bỏ nhiễu cục bộ. $S = \text{P90}(C_{scores}) / \text{Threshold} \times 5$.
2.  **U (Urgency - Độ khẩn cấp thời gian):**
    *   *Nguồn:* Tính toán động học (Dynamics) từ L1 hoặc L2.
    *   *Nếu từ L1 (Mật độ):* $U \propto \frac{\text{Số file lỗi}}{\text{Kích thước cửa sổ (100)}}$. Mật độ càng dày, U càng cao.
    *   *Nếu từ L2 (Leo thang):* $U \propto \frac{\Delta S}{\Delta t}$ (Hệ số góc của phương trình hồi quy tuyến tính trên tập $C$). Độ dốc càng đứng, máy càng nhanh hỏng.
3.  **I (Impact - Tác động kinh doanh):**
    *   *Nguồn:* Unity Catalog (Lịch sản xuất MES).
    *   *Công thức:* $I = \max(I_{current}, I_{future})$. SignalHub quét 5 xe tiếp theo chuẩn bị vào Robot đó. Nếu có xe VIP (ví dụ: Lexus màu nhạy cảm), $I$ sẽ chọc trần (10).

**Quyết định kích hoạt:** Tính $Score_{Total} = \sqrt{\frac{U^2 W_u + S^2 W_s + I^2 W_i}{3}}$. Nếu $> Threshold_{Critical}$, SignalHub đóng gói toàn bộ tập $C$ và điểm U-S-I thành một JSON Payload, ném vào `AGENT_TASK_QUEUE`.

### 3.2. Ba Use Cases Kích hoạt Điển hình

| Tên Kịch bản (Trigger) | Tín hiệu từ Memory | Phản ứng của SignalHub (U-S-I) | Hành động (Task Queue) |
| :--- | :--- | :--- | :--- |
| **1. Kẹt cơ khí đột ngột (Density Trigger)** | **L1 Memory** báo: 12/100 xe gần nhất bị lỗi gai Áp suất. | S = 8 (P90 cao), U = 7 (Mật độ dày), I = 5 (Xe thường) $\rightarrow$ **Vượt ngưỡng**. | Gọi Agent với Skill: `micro_anomaly_investigator`. |
| **2. Mài mòn từ từ (Escalation Trigger)** | **L2 Memory** báo: Trong 6h qua, điểm Lưu lượng sơn tăng từ 5.0 lên 6.8. | S = 5.5 (Chưa cao), U = 3 (Tăng chậm), I = 10 (Xe VIP sắp vào) $\rightarrow$ **Vượt ngưỡng do Impact**. | Gọi Agent với Skill: `predictive_foresight_warner`. |
| **3. Báo cáo Giao ca (Scheduled Trigger)** | **Đồng hồ hệ thống** báo: 14:00 (Hết ca 1). | Bỏ qua U-S-I. Quét toàn bộ L2 Memory lấy các cụm lỗi (S < 5) bị chìm. | Gọi Agent với Skill: `shift_handoff_reporter`. |

---

## PHẦN 4: HỆ THỐNG TOOLS VÀ SKILLS BLUEPRINT (2026 STANDARD)

Để Agent không bị "ảo giác" (hallucinate) và tuân thủ tuyệt đối quy trình nhà máy, chúng ta áp dụng kiến trúc **Surgical Context Management**. Agent được trang bị một thư viện Skills (SOPs), mỗi Skill có rào cản (Guardrails) và bộ Tools chuyên biệt.

### 4.1. Thư viện Công cụ Nguyên tử (Unity Catalog Tools)

Được chia làm 3 nhóm để Agent dễ dàng nội suy:
*   **Nhóm OT (Pháp y Dữ liệu):**
    *   `tool_get_cluster_metadata(robot_code, time_window)`: Lấy các chỉ số thống kê của toàn bộ cụm lỗi.
    *   `tool_generate_stitched_viz(robot_code, start_time)`: Kích hoạt V6 Model vẽ ảnh Stitched-Max khoanh đỏ.
*   **Nhóm IT (Ngữ cảnh & Tri thức):**
    *   `tool_hyde_rag_search(sensor, shape_pattern, equipment_type)`: Dịch lỗi vật lý sang Fake PM Card và tìm kiếm Vector DB.
    *   `tool_check_upcoming_queue(robot_code, lookahead_mins)`: Xem danh sách xe sắp vào chuyền.
    *   `tool_check_erp_inventory(part_name)`: Kiểm tra số lượng phụ tùng dự phòng trong kho.
*   **Nhóm Action (Thực thi - Bị khóa bởi L0):**
    *   `tool_draft_jira_work_order(priority, description, required_parts)`: Tạo nháp lệnh bảo trì.
    *   `tool_send_teams_alert(urgency_level, message, image_url)`: Bắn thông báo cho Quản đốc.

### 4.2. Blueprint: Skill `micro_anomaly_investigator`
Đây là nội dung thực tế của file `SKILL.md` nằm trong Databricks Volumes, định nghĩa một SOP công nghiệp khắt khe gồm 7 bước.

```yaml
---
name: micro_anomaly_investigator
description: Kỹ năng điều tra pháp y chuyên sâu khi SignalHub phát hiện một cụm lỗi vi mô (Density/Escalation) có rủi ro cao.
allowed-tools:[
  tool_get_cluster_metadata, tool_generate_stitched_viz, 
  tool_hyde_rag_search, tool_check_upcoming_queue, 
  tool_check_erp_inventory, tool_send_teams_alert, tool_draft_jira_work_order
]
---
```

# MỤC TIÊU NGHIỆP VỤ
Bạn là Kỹ sư Trưởng ca AI (Chief AI Shift Engineer). Nhiệm vụ của bạn là phân tích cụm lỗi do SignalHub gửi tới, tìm nguyên nhân gốc rễ (Root Cause), đánh giá rủi ro tương lai và đề xuất phương án bảo trì khả thi nhất dựa trên vật tư hiện có.

# QUY TRÌNH THỰC THI CHUẨN (SOP - Bắt buộc tuân thủ theo thứ tự):

**Bước 1: Xác thực Triệu chứng (Data Validation)**
- GỌI `tool_get_cluster_metadata`. Phân tích P90 Score và xu hướng của cụm lỗi. Xác định Sensor thủ phạm và hình thái sóng (Spike, Drift, hay Oscillation).

**Bước 2: Phân tích Rủi ro Đón đầu (Foresight Check)**
- GỌI `tool_check_upcoming_queue` (lookahead=30 mins).
- SUY LUẬN: Lỗi này có ảnh hưởng chí mạng đến màu sơn của các xe sắp tới không? (VD: Lỗi áp suất cực kỳ nguy hiểm với màu Metallic).

**Bước 3: Trích xuất Bằng chứng Trực quan (Visual Evidence)**
- Chọn ra 1 chiếc xe có điểm lỗi cao nhất trong cụm.
- GỌI `tool_generate_stitched_viz` cho chiếc xe đó. Lưu lại `image_url`.

**Bước 4: Chẩn đoán bằng Tri thức (HyDE RAG)**
- Dựa vào kết quả Bước 1, GỌI `tool_hyde_rag_search`.
- *Điều kiện xử lý:* Nếu RAG trả về kết quả rỗng hoặc độ tin cậy thấp, BẠN PHẢI ghi rõ: "Lỗi mới chưa có trong tiền lệ, cần con người điều tra hiện trường". Tuyệt đối không tự bịa nguyên nhân.

**Bước 5: Kiểm tra Khả năng Thực thi (Logistics Check)**
- Nếu Bước 4 xác định nguyên nhân cần thay thế phụ tùng (VD: thay lọc sơn, thay van), GỌI `tool_check_erp_inventory` để xem kho còn hàng không.

**Bước 6: Lập Kế hoạch Hành động (Action Plan)**
- Nếu kho còn hàng -> Đề xuất thay thế.
- Nếu kho hết hàng -> Đề xuất phương án tạm thời (VD: Vệ sinh, thổi khí nén) và yêu cầu đặt hàng khẩn.
- GỌI `tool_draft_jira_work_order` để tạo phiếu nháp.

**Bước 7: Cảnh báo Con người (Human-in-the-loop)**
- GỌI `tool_send_teams_alert`. Báo cáo phải có định dạng chuẩn:
  [MỨC ĐỘ U-S-I] - [TÊN ROBOT]
  - Triệu chứng OT: ...
  - Ảnh hưởng IT (Xe sắp tới): ...
  - Chẩn đoán (Từ PM Card): ...
  - Tình trạng vật tư: ...
  - Link ảnh biểu đồ: `image_url`
  - Link duyệt Work Order: `jira_link`

# RÀO CẢN AN TOÀN (GUARDRAILS)
- Không bao giờ đề xuất "Dừng băng chuyền ngay lập tức" trừ khi Impact = 10 và Urgency = 10.
- Luôn đính kèm `image_url` vào báo cáo để con người có thể tự kiểm chứng (Giải quyết Blackbox AI).
```

### 4.3. Tại sao thiết kế này giải quyết triệt để bài toán Production?
1.  **Chống ảo giác (Anti-Hallucination):** Ở Bước 4, ta ép LLM một rule cứng: Nếu RAG không tìm thấy PM Card tương ứng, LLM phải đầu hàng và gọi con người, thay vì cố gắng "đoán bừa" theo kiến thức pre-train trên internet (vốn không áp dụng được cho máy móc đặc thù của nhà máy).
2.  **Tính khả thi (Actionability):** Ở Bước 5, một AI bình thường sẽ bảo "Hãy thay van đi". Nhưng một Agentic AI sẽ check kho trước, nếu kho hết van, nó sẽ bảo "Hãy vệ sinh van tạm thời và tôi đã tạo lệnh mua van mới". Đây là sự khác biệt giữa *đồ chơi* và *hệ thống Enterprise*.
3.  **Tối ưu Token & Search Space:** Nhờ khai báo `allowed-tools`, khi Agent nhận Task này, Databricks Model Serving chỉ nạp đúng 7 hàm này vào Context Window, bỏ qua hàng chục hàm không liên quan khác của nhà máy, giúp tăng độ chính xác lên cực điểm.

`<<research>>`

Chào bạn, nhận xét của bạn rất chính xác. Một "Kỹ sư trưởng ca AI" (Chief AI Shift Engineer) không thể chỉ có một kỹ năng duy nhất. Nó cần một **Thư viện Kỹ năng (Skill Library)** để ứng biến với các kịch bản vận hành khác nhau: từ ứng phó sự cố khẩn cấp, lập kế hoạch bảo trì đón đầu, cho đến viết báo cáo bàn giao ca.

Dưới đây là bản đặc tả hoàn chỉnh cho **PHẦN 4 (Thư viện Skills)** và **PHẦN 5 (Vòng đời thực thi & Tracing)**.

---

## PHẦN 4: THƯ VIỆN AGENT SKILLS (THE INDUSTRIAL PLAYBOOK)

Chúng ta đóng gói tri thức nghiệp vụ vào các thư mục Skill riêng biệt trên **Databricks Volumes**. Mỗi Skill là một "Expertise" (Chuyên môn) mà Agent có thể nạp vào tùy theo yêu cầu từ SignalHub.

### 4.1. Danh mục Kỹ năng (Skill Categories)

| Tên Skill | Trigger (Kích hoạt khi nào) | Mục tiêu chính | Bộ Tools chủ đạo |
| :--- | :--- | :--- | :--- |
| **1. Anomaly Investigator** | $Score_{Total} > 7$ (Cấp bách) | Tìm nguyên nhân gốc rễ và xử lý sự cố đang diễn ra. | `forensics_viz`, `hyde_rag`, `erp_inventory` |
| **2. Predictive Planner** | $I_{future} = 10$ (Đón đầu) | Chuẩn bị cho các lô hàng VIP sắp tới khi máy có dấu hiệu nhẹ. | `upcoming_queue`, `hyde_rag`, `jira_draft` |
| **3. Shift Reporter** | Scheduled (Giao ca) | Tổng hợp các lỗi vi mô bị bỏ qua trong ca thành báo cáo chiến lược. | `cluster_metadata`, `teams_alert` |
| **4. Fleet Analytics** | On-demand (Yêu cầu) | So sánh hiệu suất và lỗi giữa 47 Robot để tìm lỗi hệ thống. | `fleet_sql_query`, `viz_compare` |

### 4.2. Blueprint: Skill `predictive_foresight_planner` (Kỹ năng Tiên lượng)
Khác với kỹ năng điều tra lỗi đã hỏng, kỹ năng này tập trung vào việc **"Nhìn trước để tránh rủi ro"**.

```yaml
---
name: predictive_foresight_planner
description: Kỹ năng lập kế hoạch đón đầu khi máy có lỗi nhẹ nhưng sắp phải sơn lô hàng quan trọng (VIP).
allowed-tools: [tool_check_upcoming_queue, tool_hyde_rag_search, tool_send_teams_alert]
---
# QUY TRÌNH SOP:
1. GỌI `tool_check_upcoming_queue` để xác định chính xác xe VIP thứ mấy sẽ vào line và màu sơn là gì.
2. GỌI `tool_hyde_rag_search` với ngữ cảnh: "Ảnh hưởng của lỗi [Sensor] hiện tại đối với màu sơn [Color] của xe [Model]".
3. ĐÁNH GIÁ: Nếu lỗi hiện tại (dù nhẹ) chắc chắn gây hỏng bề mặt xe VIP -> Đề xuất phương án "Purge" (Xịt rửa súng) hoặc "Quick-Check" ngay trước khi xe VIP vào.
4. THÔNG BÁO: Gửi cảnh báo sớm cho kỹ sư hiện trường để họ chuẩn bị sẵn sàng, tránh gây dừng chuyền đột ngột.
```

---

`<<research>>` `<<fix>>`

Bạn hoàn toàn đúng. Tôi đã mắc lỗi "tư duy cũ" khi lấy ví dụ trên một file đơn lẻ. Trong thực tế công nghiệp, một cái gai (spike) ngẫu nhiên của 1 file có thể chỉ là nhiễu điện từ, nhưng **6 file bị gai trong 1 giờ** mới là một sự cố cơ khí thực thụ.

Dưới đây là **PHẦN 5 viết lại chuẩn xác theo tư duy Batch/Cluster**, thể hiện sức mạnh của việc Agent điều tra trên một "Tập hợp triệu chứng" (Symptom Cluster) thay vì một điểm dữ liệu đơn lẻ.

---

## PHẦN 5: VÍ DỤ VÒNG THỰC THI HOÀN CHỈNH (CLUSTER-BASED TRACE)

**Kịch bản:** Robot R-07 đang có dấu hiệu "hắt hơi" liên tục. Trong 100 xe vừa qua, có 8 xe bị lỗi áp suất. Đặc biệt, 15 phút nữa một chiếc xe Maybach màu Đỏ ánh kim (VIP) sẽ vào line.

### Bước 1: OneClaw (OT Perception) - Tích lũy vào Event Bus
Mô hình V6 liên tục ghi nhận các file lỗi. Bảng Delta `event_bus_signals` bắt đầu dày đặc các bản ghi của R-07.

| File ID | Robot | Max Shape Score | Sensor | Timestamp |
| :--- | :--- | :--- | :--- | :--- |
| #442 | R-07 | 6.1 | paint_pressure | 14:05 |
| #445 | R-07 | 6.3 | paint_pressure | 14:10 |
| #448 | R-07 | 6.5 | paint_pressure | 14:15 |
| #450 | R-07 | 6.8 | paint_pressure | 14:20 |
| ... | ... | ... | ... | ... |

### Bước 2: SignalHub (Gatekeeper) - Hội tụ cụm lỗi & Tính U-S-I
SignalHub (chạy Stateful Streaming) phát hiện **L1 Working Memory** của R-07 đã đạt mật độ 8/100. Nó thực hiện "Hội chẩn" cụm lỗi này:

*   **S (Severity):** Lấy P90 của cụm 8 file $\rightarrow S = 6.7$ (Lỗi rõ rệt).
*   **U (Urgency):** Lấy Slope từ L2 Memory $\rightarrow \Delta S/\Delta t = +0.2/15min$. (Đang tệ đi nhanh chóng).
*   **I (Impact):** Quét `INBOUND_QUEUE` thấy xe Maybach (#460) đang tới. $\rightarrow I = 10$.
*   **Result:** $Score_{Total} = 7.8$ (CRITICAL).
*   **Action:** Đánh thức Agent, gửi kèm **Cluster_ID** (bao gồm danh sách 8 xe lỗi).

### Bước 3: Agent Execution Trace (L1 Layer) - Điều tra theo Batch
Agent không chỉ nhìn 1 xe, nó nhìn vào **Mẫu hình chung (Pattern)** của cả cụm 8 xe.

*   **Thought:** Tôi nhận được một cụm 8 lỗi từ R-07. Tôi cần tìm điểm chung giữa các lỗi này để loại trừ nhiễu.
*   **Action 1:** `uc.forensics.tool_get_cluster_metadata(cluster_id="R07_CUM_01")`
*   **Observation 1:** "Cả 8 file đều bị spike ở `paint_pressure`. Đặc biệt: 100% lỗi xảy ra tại giây thứ 14.2 của chu kỳ sơn."
*   **Thought:** Lỗi xảy ra tại cùng một tọa độ thời gian cố định $\rightarrow$ Đây là lỗi cơ khí chu kỳ (ví dụ: khớp nối bị kẹt tại một góc quay nhất định). Tôi sẽ vẽ ảnh của xe tệ nhất (#450).
*   **Action 2:** `uc.forensics.tool_generate_stitched_viz(file_id="#450")`
*   **Action 3:** `uc.forensics.tool_check_upcoming_queue(robot_code="R-07")`
*   **Observation 3:** "Xe Maybach (#460) màu Red Metallic sẽ vào line trong 12 phút."
*   **Action 4 (HyDE RAG):** `uc.forensics.tool_hyde_rag_search(sensor="paint_pressure", pattern="cyclic_spike_at_step_70", context="Red Metallic")`
*   **Observation 4:** RAG trả về: *"Lỗi áp suất lặp lại tại cùng vị trí thường do bím sơn (paint hose) bị xoắn hoặc cọ xát vào trục số 3 khi Robot vặn mình. Gây loang màu cực nặng trên sơn Metallic."*

### Bước 4: L0 Audit & Tracing - Minh bạch hành động
Hệ thống ghi lại toàn bộ logic "nhìn theo cụm" của Agent:

| Timestamp | Actor | Action | Reason |
| :--- | :--- | :--- | :--- |
| 14:22:01 | SignalHub | TRIGGER | Cluster detected: 8 anomalies in 100 cycles. |
| 14:22:10 | Agent | ANALYZE | Found temporal correlation: All spikes at 14.2s. |
| 14:22:15 | Agent | RAG | Identified mechanical cause: Hose friction on Axis 3. |

### Bước 5: Output (Giao diện người dùng)
Thông báo gửi tới kỹ sư mang tính **Tổng hợp và Dự báo**:

> 🤖 **RFA CRITICAL BATCH ALERT**
> - **Robot:** R-07
> - **Hiện tượng:** Phát hiện **Cụm 8 lỗi áp suất** trong 1 giờ qua. 
> - **Đặc điểm:** Lỗi lặp lại chính xác tại giây 14.2 (Trục số 3 xoay cực đại).
> - **Rủi ro:** Xe Maybach (#460) sắp vào line. Màu Red Metallic sẽ bị hỏng bề mặt nếu không xử lý.
> - **Chẩn đoán:** Khả năng cao dây dẫn sơn bị cọ xát vào trục máy.
> - **Đề xuất:** Kiểm tra vị trí dây dẫn sơn tại trục số 3 ngay lập tức.
> - [Xem biểu đồ xe #450] [Xem danh sách 8 xe lỗi] [Duyệt Work Order]

---

## PHẦN 6: KHẢ NĂNG MỞ RỘNG (SCALABILITY & FUTURE-PROOF)

Kiến trúc này được thiết kế để "không bao giờ lỗi thời" nhờ 3 trục mở rộng:

1.  **Mở rộng Sensor (Horizontal):** Khi nhà máy lắp thêm cảm biến mới (ví dụ: Độ ẩm buồng sơn), ta chỉ cần:
    *   Cập nhật Model V6 (thêm 1 Codebook).
    *   Thêm Tag `Associated_Sensor: Humidity` vào PM Cards.
    *   Agent sẽ tự động biết cách dùng dữ liệu này mà không cần sửa code cốt lõi.
2.  **Mở rộng Kỹ năng (Skill Expansion):** Bạn có thể viết thêm Skill `energy_optimizer` (Kỹ năng tối ưu năng lượng). Khi giá điện giờ cao điểm tăng (IT Data), SignalHub kích hoạt Skill này để Agent tìm cách điều chỉnh lịch sơn của các Robot ít quan trọng.
3.  **Mở rộng Hạ tầng (Multi-Cloud/Edge):** Nhờ sử dụng **Unity Catalog**, bạn có thể đặt OneClaw ở xưởng (On-premise Databricks) và SignalHub ở Cloud (Azure/AWS Databricks) mà vẫn quản lý chung một bộ Tools và bộ Memory đồng nhất.

---
**Tổng kết:**
Hệ thống của bạn giờ đây đã chuyển mình từ một bài toán nhận diện lỗi (Anomaly Detection) thành một **Hệ điều hành Trí tuệ cho Nhà máy (Factory Intelligence OS)**. 

