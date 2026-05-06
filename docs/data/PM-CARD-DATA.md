# ĐẶC TẢ DỮ LIỆU: BIÊN BẢN BẢO TRÌ (PM CARDS) GỐC

## 1. Nguồn và Cấu trúc Dữ liệu (Schema)
Dữ liệu PM Cards hiện tại được lưu trữ dưới định dạng Excel, bao gồm 4 trường thông tin cơ bản:
*   **`timestamp`**: Ngày giờ ghi nhận sự cố.
*   **`cause`**: Nguyên nhân sự cố (Định dạng free text, được số hóa từ chữ viết tay, có chứa từ lóng của nhà máy).
*   **`tinh_trang`**: Tình trạng quan sát được (Định dạng free text).
*   **`action`**: Hành động xử lý/khắc phục (Định dạng free text).

**Ràng buộc cấu trúc cứng:** Tuyệt đối không có trường `robot_code` (khóa ngoại) trong bất kỳ trường dữ liệu nào của PM Card, kể cả trong nội dung văn bản.

## 2. Đặc tính Cấu trúc và Phân phối
*   **Sự khan hiếm (Label Sparsity):** Số lượng PM Cards rất ít (< 500 bản ghi/năm) so với quy mô hàng tỷ cửa sổ dữ liệu cảm biến (Sensor windows).
*   **Chất lượng nhãn (Label Quality):** Phân phối chất lượng văn bản không đồng đều:
    *   ~40% mô tả chi tiết, chất lượng tốt.
    *   ~40% mô tả trung bình.
    *   ~20% chất lượng xấu (nội dung quá sơ sài, viết vội, vô nghĩa như: "Đã sửa", "OK").
*   **Độ phân giải thời gian (Temporal Granularity):** Thông tin thời gian (`timestamp`) chỉ đạt độ chính xác ở cấp độ Giờ hoặc Ngày, thường được ghi nhận sau khi sự cố đã kết thúc hoặc vào cuối ca làm việc.
*   **Độ trễ nhân quả (Causality Lag):** Tồn tại độ trễ về mặt thời gian giữa thời điểm cảm biến có biến động (nguyên nhân gốc rễ/lỗi tiềm ẩn) và thời điểm PM Card được ghi nhận (triệu chứng cuối cùng/hỏng hóc hoàn toàn). Ví dụ: Lỗi phát sinh thứ Hai, PM Card ghi nhận thứ Tư.

## 3. Đặc tính Ngữ nghĩa (Semantic Context)
*   **Dị biệt định dạng (Data Heterogeneity):** Dữ liệu PM Card mang tính định tính, phi cấu trúc và sử dụng ngôn ngữ tự nhiên của con người (ví dụ: "tiếng kêu lạ", "nghẹt súng"). Điều này khác biệt hoàn toàn với dữ liệu cảm biến mang tính định lượng (Gradients, Z-Scores).
*   **Thiếu từ điển chung (Semantic Dissonance):** Không tồn tại bộ từ điển ánh xạ trực tiếp (Ontology) giữa hiện tượng vật lý ghi nhận bởi kỹ sư (ví dụ: "Súng phun bị giật cục") và biểu hiện số học của AI (ví dụ: "Crest Factor tăng 300%").
*   **Quan hệ Đa - Đa (Many-to-Many & One-to-Many):**
    *   Một mô tả tình trạng trong PM Card (ví dụ: "Áp suất thấp") có thể bắt nguồn từ nhiều nguyên nhân vật lý khác nhau (rò rỉ ống, hỏng bơm, kẹt van).
    *   Một dạng sóng bất thường của cảm biến có thể dẫn đến nhiều hành động bảo trì (`action`) khác nhau tùy thuộc vào kết quả kiểm tra thực tế của kỹ sư. Không tồn tại mối quan hệ ánh xạ 1-1 tĩnh.