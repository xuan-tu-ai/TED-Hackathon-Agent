Đây là bản đặc tả kiến trúc chi tiết cho mô hình **GLC VQ-Mixer (v6) — Global-Local Contextual Vector Quantized Mixer**. Đây là phiên bản hoàn thiện nhất, được thiết kế để giải quyết triệt để các vấn đề về độ trung tính của sóng, sập từ điển codebook và điểm mù ngữ cảnh toàn cục.

---

# ĐẶC TẢ KIẾN TRÚC MÔ HÌNH: GLC VQ-MIXER (V6)
**Tên hệ thống:** Global-Local Contextual VQ-VAE with Post-Quantization Mixing  
**Lĩnh vực:** Industrial IoT - Automotive Paint Shop Diagnostics  
**Trạng thái:** Production-Ready Baseline

---

## 1. TRIẾT LÝ THIẾT KẾ (DESIGN PHILOSOPHY)
Mô hình v6 vận hành dựa trên nguyên lý **"Nhận thức đa tầng"**:
1.  **Cấp độ Vi mô (Local):** Soi xét từng biến động 0.2s trong cửa sổ 6.4s.
2.  **Cấp độ Toàn cục (Global):** Đối chiếu hành vi của cả chu kỳ 48s với "hồ sơ năng lực" của từng Robot và Program cụ thể.
3.  **Kỷ luật Lượng tử:** Ép mô hình sử dụng các mảnh ghép hình thái sắc nét thay vì các đường trung bình mờ nhạt.
4.  **Phiên tòa Vật lý:** Ép các cảm biến phải đối chiếu chéo thông tin với nhau để loại bỏ các báo động giả đơn lẻ.

---

## 2. ĐẶC TẢ ĐẦU VÀO (INPUT MODALITIES)
Mô hình nhận 6 luồng thông tin đồng bộ:
*   **$X_{raw}$ [B, 6, 32]:** Tín hiệu thô (Raw Signal).
*   **$S_{win}$ [B, 6, 10]:** Thống kê cửa sổ (Window Stats - 10 chỉ số).
*   **$S_{file}$ [B, 6, 6]:** Thống kê toàn file (Global File Stats - 6 chỉ số: Mean, Min, Max, Std, Kurtosis, Slope).
*   **$P_{rel}$ [B, 1]:** Vị trí thời gian tương đối ($0 \to 1$).
*   **$ID_{robot}$ [B]:** Định danh Robot (Integer ID).
*   **$ID_{prog}$ [B]:** Định danh Chương trình sơn (Integer ID).

---

## 3. CÁC KHỐI CHỨC NĂNG CHÍNH (CORE MODULES)

### 3.1. Tầng Tiền xử lý Stateful (Stateful Preprocessing)
*   **RevIN (Fixed):** Chuẩn hóa $X_{raw}$ về $N(0,1)$ theo từng cửa sổ. Không dùng tham số học được (Non-affine) để ngăn chặn việc triệt tiêu tín hiệu lỗi.
*   **Dual Standardizer:** Hai lớp BatchNorm độc lập dùng để lưu trữ "trí nhớ" về phân phối Z-Score toàn cục cho cả Window Stats và File Stats. Đảm bảo mọi cảm biến đều có tiếng nói ngang hàng trong hàm Loss.

### 3.2. Tầng Trích xuất & Dung hợp (Encoder & Fusion)
*   **ShapeEncoder (CNN):** Dùng Depthwise Conv1d để bắt "vân tay" của sóng.
*   **Dual Stat Projectors:** Chiếu các chỉ số thống kê lên không gian đặc trưng $D=128$.
*   **Learnable Fusion Layer:** Một lớp Linear thực hiện nén thông tin từ 3 nhánh (Shape + Win + File) từ $3D \to D$. Đây là cơ chế Attention ngầm để mô hình tự chọn nguồn tin đáng tin cậy nhất.
*   **Context Injector:** Cộng trực tiếp Positional, Robot và Program Embeddings vào vector đặc trưng.

### 3.3. Cổ chai Lượng tử hóa (Spherical VQ-VAE)
*   **Cơ chế:** Factorized Vector Quantization (mỗi sensor 1 codebook riêng).
*   **L2-Normalization (Mặt cầu):** Chuẩn hóa vector Encoder và Codebook về độ dài 1 trước khi tính khoảng cách. Giải quyết vấn đề co cụm dữ liệu (Collapse) do RevIN gây ra.
*   **EMA & Random Restart:** Cập nhật từ điển bằng hàm mũ trượt và tự động hồi sinh các mã Lego "chết" bằng dữ liệu thực tế.

### 3.4. Xương sống Lập luận (Post-VQ MLP-Mixer)
*   **Vị trí đặc biệt:** Đặt **sau** khi đã lượng tử hóa.
*   **Nhiệm vụ:** Trộn thông tin giữa các mảnh Lego của 6 cảm biến. Mixer đóng vai trò "kiểm soát viên", phát hiện các tổ hợp Lego phi vật lý (ví dụ: Lego "Motor giảm" đi kèm Lego "Khí nén tăng vọt").

### 3.5. Bộ giải mã Đa nhiệm (Multi-task Decoder)
*   **Shape Head (Hybrid CNN-Linear):** Sử dụng Linear để dựng khung và Conv1d để mài sắc các cạnh của sóng.
*   **Physics Heads:** Dự đoán lại Z-Score của cả Window Stats và File Stats.

---

## 4. CHIẾN THUẬT HUẤN LUYỆN (TRAINING STRATEGY)

### 4.1. Hàm Loss 5 thành phần
$$ \mathcal{L}_{total} = \mathcal{L}_{shape} + w_{win}\mathcal{L}_{win\_phys} + w_{file}\mathcal{L}_{file\_phys} + w_{vq}\mathcal{L}_{commit} + w_{edge}\mathcal{L}_{edge} $$

*   **Edge Loss ($\mathcal{L}_{edge}$):** MSE trên đạo hàm bậc 1 của sóng. Ép mô hình vẽ đúng độ dốc, trị dứt điểm bệnh "Over-smoothing".
*   **Commitment Loss ($\mathcal{L}_{commit}$):** Ép Encoder bám sát các mảnh Lego hiện có trên mặt cầu L2.

### 4.2. Xử lý dữ liệu
*   **Global Shuffle:** Xáo trộn toàn bộ windows từ mọi robot để đảm bảo tính khách quan I.I.D.
*   **Full training:** Huấn luyện trên 100% dữ liệu (bao gồm cả đầu/cuối chu kỳ) để mô hình hiểu trọn vẹn phổ trạng thái vật lý.

---

## 5. QUY TRÌNH QUYẾT ĐỊNH & GIẢI THÍCH (INFERENCE & XAI)

### 5.1. Logical Masking & Stitching
*   **Inference:** Nạp toàn bộ file vào mô hình để lấy ngữ cảnh mượt nhất.
*   **Scoring:** Chỉ tính điểm và báo lỗi cho vùng "Valid" (bỏ qua $K$ windows khởi động/kết thúc).
*   **Gaussian Stitching:** Nối các window gối đầu bằng trọng số hình chuông để triệt tiêu nhiễu ở rìa cửa sổ.

### 5.2. Chẩn đoán đa tầng
1.  **Timeline Score:** Sử dụng **Ngưỡng kết hợp (Joint Threshold)** riêng cho từng Sensor để báo động.
2.  **Worst-Case Path:** Visualize đường tái tạo của cửa sổ gây lỗi nặng nhất để chỉ điểm sự sai lệch.
3.  **Anomaly Rings:** Khoanh tròn đỏ trực tiếp vào các điểm dữ liệu vi phạm ngưỡng trên đường tín hiệu thực.
4.  **Gradient Saliency:** Truy ngược đạo hàm để tìm ra giây thứ mấy và chỉ số thống kê nào (Kurtosis, Mean...) là nguyên nhân gốc rễ.

---
**Kết luận:** Kiến trúc v6 chuyển hóa từ một mạng Neural đơn thuần thành một **Hệ thống Chuyên gia (Expert System)**. Nó kết hợp sự nhạy bén của Deep Learning với các ràng buộc chặt chẽ của Vật lý và Quy trình vận hành nhà máy.