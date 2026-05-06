# TÀI LIỆU CHUYỂN GIAO DỰ ÁN: PAINTING ABNORMAL AGENT
## 1. BỐI CẢNH & BÀI TOÁN (CONTEXT)
*   **Môi trường:** Xưởng sơn ô tô (Automotive Paint Shop) với 47 Robot vận hành song song.
*   **Dữ liệu đầu vào:** 
    *   **OT Data:** Log cảm biến tần suất cao (5Hz - 0.2s/mẫu). Gồm 6 kênh chính: Tốc độ motor (Air Motor Rotation Speed), Khí nén (Air Motor Pneumatic Output Value Bit), Dòng điện/Điện áp cao áp (High Pressure Voltage/Current), Áp suất (Paint Pressure) và Lưu lượng sơn (Paint Flow rate (cc per min)).
    *   **IT Data:** Biên bản bảo trì (PM Cards) dạng văn bản tự nhiên, chứa tri thức về các sự cố quá khứ.
*   **Mục tiêu:** Xây dựng hệ thống "Pháp y dữ liệu" (Forensics) giúp phát hiện sớm các lỗi vi mô (Micro-anomalies) và tự động giải thích nguyên nhân dựa trên quy trình chuẩn.
## 2. THÁCH THỨC KỸ THUẬT (THE CORE DIFFICULTIES)
1.  **Pha loãng tín hiệu (Signal Dilution):** Lỗi chỉ kéo dài 0.2s - 0.6s (1-3 điểm ảnh) trong chu kỳ 48 giây. Các phương pháp thống kê truyền thống dễ dàng bỏ sót.
2.  **Biến thiên Baseline:** Mỗi Robot và mỗi loại xe (Program) có dải vận hành khác nhau. Không thể đặt một ngưỡng cứng cho toàn nhà máy.
3.  **Dữ liệu không nhãn (Unlabeled Data):** Phải học hoàn toàn theo phương pháp tự giám sát (Self-supervised), tự định nghĩa thế nào là "Bình thường".
4.  **Độ trung tính của VAE:** Các mô hình VAE liên tục có xu hướng "làm mượt" các cạnh sắc (xung vuông), dẫn đến việc báo động giả tại các điểm chuyển trạng thái.

## 3. KIẾN TRÚC MODEL DEEPLEARNING đã triển khai
Chúng ta đã vượt qua 5 phiên bản R&D để đi đến kiến trúc **GLC VQ-Mixer (V6)**:

Xem chi tiết ở [GLC VQ-Mixer (V6)](./references/DL_MODEL.md)

## 4. CÁC CỘT MỐC ĐÃ ĐẠT ĐƯỢC (ACCOMPLISHMENTS)
1.  **Hội tụ mô hình:** Đã huấn luyện thành công Model v5 trên Databricks GPU sử dụng **Mosaic Streaming (MDS)**, giải quyết triệt để lỗi tràn bộ nhớ (OOM).
2.  **Hiệu chuẩn ngưỡng (Joint Calibration):** Phát triển thuật toán tìm cặp ngưỡng $(x, y)$ tối ưu trên tập Validation sao cho tổng tỷ lệ lỗi (Shape OR Physics) khớp với mục tiêu kinh doanh (ví dụ P99.5).
3.  **Hệ thống Giải thích (XAI):**
    *   **Stitched-Max Visualization:** Nối chuỗi các window bằng trọng số Gaussian và khoanh vùng lỗi bằng vòng tròn đỏ (Red Rings).
    *   **Gradient Saliency:** Heatmap chỉ điểm chính xác giây thứ mấy và cảm biến nào gây lỗi.
    *   **Weight Analysis:** Trích xuất tương quan hệ thống từ trọng số mô hình để hiểu sự liên đới giữa các bộ phận.
## 5. LỘ TRÌNH TIẾP THEO (FUTURE ROADMAP)
1.  **Knowledge Linking:** Xây dựng cơ chế ánh xạ (Input Mapping) giữa đầu ra AI và kho PM Cards. Sử dụng "Dấu vân tay Lego" (Indices từ VQ-VAE) để tìm kiếm sự cố tương tự trong quá khứ.
2.  **Agentic Cookbook:** Phát triển AI Agent 3 cấp độ để tự động hóa quy trình chẩn đoán (SOP). Agent sẽ chọn "Cookbook" phù hợp dựa trên triệu chứng và lịch sử bảo trì.
3.  **VQ-VAE Semantic Mapping:** Gán nhãn ngữ cảnh cho từng mảnh Lego trong Codebook (ví dụ: Lego #42 = "Sụt áp nhanh") để AI có thể "đọc tên" lỗi bằng ngôn ngữ kỹ thuật.

Đọc tài liệu sau. DỰ ÁN: AI CHO ROBOT CÔNG NGHIỆP