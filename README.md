# Datathon-BIUS

# DATATHON-BIUS — Datathon 2026 Round 1

Repository này chứa toàn bộ mã nguồn, notebook phân tích và pipeline mô hình cho bài thi **DATATHON 2026 — The Gridbreakers, Round 1**. Dự án tập trung vào bài toán phân tích hoạt động kinh doanh của một doanh nghiệp thời trang thương mại điện tử tại Việt Nam và dự báo doanh thu trong giai đoạn test theo yêu cầu của Ban Tổ Chức.

## 1. Mục tiêu dự án

Dự án được xây dựng nhằm đáp ứng ba phần chính của đề thi:

1. **Multiple Choice Questions**: Tính toán trực tiếp từ dữ liệu được cung cấp để trả lời các câu hỏi trắc nghiệm.
2. **Exploratory Data Analysis & Business Insight**: Khai thác dữ liệu đa bảng để tìm ra các pattern có ý nghĩa kinh doanh, đặc biệt liên quan đến doanh thu, biên lợi nhuận, khuyến mãi, hành vi mua hàng, web traffic và vận hành.
3. **Sales Forecasting**: Xây dựng pipeline dự báo cột `Revenue` cho giai đoạn test, đảm bảo tính tái lập, kiểm soát leakage và có phần giải thích mô hình.

Dự án không sử dụng dữ liệu bên ngoài. Tất cả đặc trưng được tạo từ các file CSV do Ban Tổ Chức cung cấp.

## 2. Cấu trúc repository

```text
DATATHON-BIUS/
│
├── dataset/ # Chứa dữ liệu đầu vào của cuộc thi
├── notebooks/
│   ├── EDA.ipynb # Phần phân tích dữ liệu và rút ra business insights
│   └── forecast.ipynb # Xây dựng mô hình dự đoán
│   └── MultipleChoice.ipynb # Trả lời các câu hỏi trắc nghiệm ở Phần 01
│
├── src/
│   ├── baseline.py # Chứa source code baseline từ BTC (viết lại dưới file .py)
│   ├── data_loader.py
│   ├── feature_engineering.py # Functions tạo 3 đặc trưng chính
│   ├── metrics.py
│   ├── model.py # Chứa các functions huấn luyện mô hình LightGBM, và Ridge Regression
│   └── validation.py
│
├── .gitattributes
├── LICENSE
└── README.md
```

## 3. Dữ liệu sử dụng

Bộ dữ liệu mô phỏng hoạt động của một doanh nghiệp thời trang thương mại điện tử tại Việt Nam trong giai đoạn 04/07/2012 đến 31/12/2022. Các bảng dữ liệu được chia thành bốn nhóm:

### Master tables

| File             | Nội dung                                                                                 |
| ---------------- | ---------------------------------------------------------------------------------------- |
| `products.csv`   | Danh mục sản phẩm, giá bán lẻ, giá vốn, category, segment, size, color                   |
| `customers.csv`  | Thông tin khách hàng, thành phố, ngày đăng ký, nhóm tuổi, giới tính, acquisition channel |
| `promotions.csv` | Thông tin chiến dịch khuyến mãi, mức giảm, thời gian áp dụng, kênh áp dụng               |
| `geography.csv`  | Mã bưu chính, thành phố, vùng địa lý và quận/huyện                                       |

### Transaction tables

| File              | Nội dung                                                                            |
| ----------------- | ----------------------------------------------------------------------------------- |
| `orders.csv`      | Thông tin đơn hàng, ngày đặt, khách hàng, trạng thái, thiết bị, nguồn đơn           |
| `order_items.csv` | Chi tiết sản phẩm trong đơn hàng, số lượng, đơn giá sau khuyến mãi, discount, promo |
| `payments.csv`    | Thông tin thanh toán và số kỳ trả góp                                               |
| `shipments.csv`   | Thông tin vận chuyển, ngày gửi, ngày giao, phí vận chuyển                           |
| `returns.csv`     | Thông tin trả hàng, lý do trả hàng, số lượng trả, số tiền hoàn                      |
| `reviews.csv`     | Đánh giá sản phẩm sau giao hàng                                                     |

### Analytical tables

| File                    | Nội dung                                                                  |
| ----------------------- | ------------------------------------------------------------------------- |
| `sales.csv`             | Dữ liệu doanh thu ngày dùng cho huấn luyện, gồm `Date`, `Revenue`, `COGS` |
| `sample_submission.csv` | Định dạng file submission cần nộp                                         |

### Operational tables

| File              | Nội dung                                              |
| ----------------- | ----------------------------------------------------- |
| `inventory.csv`   | Ảnh chụp tồn kho cuối tháng và các chỉ số vận hành    |
| `web_traffic.csv` | Lưu lượng truy cập website theo ngày và nguồn traffic |

## 4. Hướng tiếp cận phân tích EDA

Phần EDA được thiết kế theo hướng business-first:
`Revenue` là gì, được tạo từ đâu, chất lượng nó ra sao và bị thất thoát ở đâu?
Cốt lõi cách tiếp cận:

1. Dựng một grain chung để phân tích

Chọn đơn vị nhỏ nhất là 1 dòng sản phẩm trong 1 đơn hàng, từ đó mới roll-up lên `order`, `ngày`, `tháng`.

2. Build lại doanh thu từ transaction rồi đối chiếu theo ngày

3. Tách bạch 2 lớp phân tích.

- Quy mô: doanh thu, số đơn, số khách mua, số lượng bán.
- Chất lượng: discount, gross margin, refund, conversion efficiency.

4. Dùng thời gian làm xương sống của toàn bộ EDA.
   Sau khi hiểu KPI, notebook kéo mọi thứ về trục ngày/tháng để đọc:
   mùa vụ theo tháng, theo thứ, theo vị trí ngày trong tháng, autocorrelation, ngày spike bất thường.
   Nghĩa là thời gian không chỉ là chart mở đầu, mà là khung để gắn mọi cơ chế còn lại vào.

5. Đi từ biến gần doanh thu nhất ra biến xa hơn. Ưu tiên các cơ chế “sinh doanh thu” trước, rồi mới sang các lớp “giải thích bối cảnh” và “thất thoát sau bán”.

   orders/order_items -> products -> promotions -> traffic/customers -> inventory/payments/geography -> shipping/returns/reviews.

### 4.1. Revenue, COGS và Gross Margin

Phân tích xu hướng doanh thu, giá vốn và biên lợi nhuận gộp theo thời gian để xác định giai đoạn tăng trưởng, suy giảm, mùa vụ và các điểm bất thường. Trọng tâm là đánh giá liệu tăng trưởng doanh thu có đi kèm với hiệu quả lợi nhuận hay doanh nghiệp đang đánh đổi margin để kích cầu.

### 4.2 Promotion Effectiveness

Kết hợp `promotions.csv`, `order_items.csv`, `orders.csv` và `sales.csv` để phân tích tác động của các chiến dịch khuyến mãi lên doanh thu, discount amount và gross margin. Phần này tập trung vào việc phân biệt chiến dịch tạo doanh thu bền vững với chiến dịch tạo áp lực lên profitability.

### 4.3 Online Demand & Web Traffic

Kết hợp `web_traffic.csv` với dữ liệu doanh thu theo ngày để đánh giá mối quan hệ giữa sessions, visitors, page views, bounce rate, conversion proxy và revenue. Phân tích này hỗ trợ đề xuất tối ưu kênh online và cải thiện hiệu quả chuyển đổi.

### 4.4 Customer & Order Behavior

Khai thác `customers.csv`, `orders.csv`, `order_items.csv` để phân tích hành vi mua lại, inter-order gap, giá trị đơn hàng, nguồn đơn hàng và nhóm khách hàng có đóng góp cao.

### 4.5 Returns, Reviews và Product Quality

Kết hợp `returns.csv`, `reviews.csv` và `products.csv` để nhận diện nhóm sản phẩm hoặc kích cỡ có tỷ lệ trả hàng cao, lý do trả hàng phổ biến và tác động tiềm năng đến biên lợi nhuận.

### 4.6 Inventory & Operational Risk

Sử dụng `inventory.csv` để phân tích stockout, overstock, fill rate, sell-through rate và mối liên hệ giữa vận hành tồn kho với khả năng đáp ứng nhu cầu.

## 5. Hướng tiếp cận mô hình dự báo

Bài toán forecasting yêu cầu dự báo `Revenue` và `COGS` cho giai đoạn test. Pipeline được xây dựng theo nguyên tắc:

1. Tạo forecasting frame theo ngày từ `sales.csv`.
2. Tạo đặc trưng thời gian: ngày, tháng, quý, năm, thứ trong tuần, cuối tuần, mùa vụ và các biến chu kỳ.
3. Tạo đặc trưng từ đơn hàng, sản phẩm, khuyến mãi, web traffic và tồn kho, chỉ sử dụng dữ liệu hợp lệ theo thời gian.
4. Chia validation theo thứ tự thời gian bằng `TimeSeriesSplit`
5. Huấn luyện baseline, Rigde Regression và LightGBM để so sánh.
6. Huấn luyện mô hình chính bằng LightGBM (Xây `FeatureBuilder` và `Pipeline` và train với `time series cross-validation`)
7. Đánh giá bằng MAE, RMSE và R2.
8. Giải thích mô hình bằng feature importance/SHAP.
9. Fit final model trên toàn bộ tập train hợp lệ và tạo `submission.csv` theo đúng format `sample_submission.csv`.

## 6. Nguyên tắc chống data leakage

Vì đây là bài toán dự báo theo thời gian, toàn bộ pipeline được thiết kế để tránh sử dụng thông tin tương lai. Các nguyên tắc chính:

- Không dùng `Revenue` hoặc `COGS` của tập test làm feature.
- Không dùng dữ liệu ngoài bộ dữ liệu được cung cấp.
- Khi tạo lag/rolling feature, chỉ sử dụng giá trị trong quá khứ.
- Khi cross-validation, mỗi fold chỉ fit feature engineering trên phần train của fold đó.
- Các biến tổng hợp từ transaction hoặc operational data phải được tạo sao cho không nhìn thấy thông tin sau ngày validation/test.
- Final model chỉ được train sau khi đã hoàn tất đánh giá trên các fold thời gian.

## 7. Metrics đánh giá

Mô hình được đánh giá bằng ba chỉ số chính:

### Mean Absolute Error — MAE

MAE đo sai số tuyệt đối trung bình giữa dự báo và thực tế. Chỉ số này dễ diễn giải theo đơn vị tiền tệ và là metric chính để đánh giá mức lệch trung bình của mô hình.

### Root Mean Squared Error — RMSE

RMSE phạt nặng hơn các lỗi lớn, phù hợp để kiểm tra liệu mô hình có bị sai mạnh ở các giai đoạn đột biến doanh thu hay không.

### R2 Score

R2 đo tỷ lệ phương sai của doanh thu được mô hình giải thích. R2 càng gần 1 thì mô hình càng giải thích tốt biến động của dữ liệu.

## 8. Cài đặt môi trường

Khuyến nghị sử dụng Python 3.10 trở lên.

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

pip install -U pip
pip install pandas numpy scikit-learn lightgbm optuna matplotlib seaborn shap jupyter
```

Nếu repository có file `requirements.txt`, có thể cài đặt trực tiếp bằng:

```bash
pip install -r requirements.txt
```

## 9. Hướng dẫn chạy lại kết quả

### Bước 1: Chuẩn bị dữ liệu

Đặt toàn bộ file CSV do Ban Tổ Chức cung cấp vào thư mục `dataset/`.

Kiểm tra cấu trúc thư mục:

```bash
DATATHON-BIUS/
├── dataset/
├── notebooks/
└── src/
```

### Bước 2: Chạy EDA

Mở notebook:

```bash
jupyter notebook notebooks/EDA.ipynb
```

Chạy toàn bộ notebook để tái tạo các bảng phân tích, biểu đồ và insight phục vụ báo cáo.

### Bước 3: Chạy forecasting pipeline

Mở notebook:

```bash
jupyter notebook notebooks/forecast.ipynb
```

Chạy lần lượt các phần:

1. Load data.
2. Feature engineering.
3. Time-series cross-validation.
4. LightGBM training.
5. Final model training.
6. Metrics evaluation.
7. Explainability.

### Bước 4: Tạo file submission

Sau khi chạy xong notebook forecasting, file `submission.csv` cần có đúng ba cột:

```text
Date,Revenue,COGS
```

File cần giữ nguyên thứ tự dòng như `sample_submission.csv`. Không được sort lại hoặc xáo trộn thứ tự dòng.

## 10. Team Information

Tên đội: `BIUS`

Cuộc thi: `DATATHON 2026 — The Gridbreakers`

Đơn vị tổ chức: `VinTelligence — VinUniversity Data Science & AI Club`
