from ultralytics import YOLO
import cv2
import os

# Load các mô hình
model_plate = YOLO('best.pt')         # Nhận diện biển số
model_char = YOLO('best-char.pt')     # Nhận diện ký tự

# Đọc ảnh đầu vào
img_path = 'img-test/7.jpg'
img = cv2.imread(img_path)
img_name = os.path.splitext(os.path.basename(img_path))[0]

# Phát hiện biển số
results_plate = model_plate(img)[0]

# Thư mục lưu kết quả
output_crop_path = 'cropped_plates'
os.makedirs(output_crop_path, exist_ok=True)

# Xử lý từng biển số
for i, box in enumerate(results_plate.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    plate_crop = img[y1:y2, x1:x2]
    h, w = plate_crop.shape[:2]

    # Resize nếu cần
    min_height = 200
    if h < min_height:
        scale = min_height / h
        new_w, new_h = int(w * scale), int(h * scale)
        plate_crop = cv2.resize(plate_crop, (new_w, new_h))

    # Dự đoán ký tự trong biển số
    results_char = model_char(plate_crop)[0]

    # Lưu danh sách ký tự kèm tọa độ
    char_list = []
    for box in results_char.boxes:
        if float(box.conf) < 0.5:
            continue
        x1c, y1c, x2c, y2c = map(int, box.xyxy[0].tolist())
        label = model_char.names[int(box.cls[0])]
        if label == "license plates":
            continue  # 👉 Bỏ qua nếu là 'license plates'
        x_center = (x1c + x2c) // 2
        y_center = (y1c + y2c) // 2
        char_list.append((x_center, y_center, label))

    if not char_list:
        continue

    # 👉 Tách dòng theo y
    y_values = [y for _, y, _ in char_list]
    y_sorted = sorted(y_values)
    y_diff = [y_sorted[i+1] - y_sorted[i] for i in range(len(y_sorted)-1)]

    threshold = 20
    line_break = 0
    for j, diff in enumerate(y_diff):
        if diff > threshold:
            line_break = (y_sorted[j] + y_sorted[j+1]) // 2
            break

    # Tách dòng trên/dưới theo y
    if line_break:
        top_line = [(x, y, c) for x, y, c in char_list if y < line_break]
        bottom_line = [(x, y, c) for x, y, c in char_list if y >= line_break]
    else:
        top_line = char_list
        bottom_line = []

    # Sắp xếp và tạo chuỗi ký tự
    top_line.sort(key=lambda x: x[0])
    bottom_line.sort(key=lambda x: x[0])

    top_text = ''.join([c for _, _, c in top_line])
    bottom_text = ''.join([c for _, _, c in bottom_line])


    # 👉 Format kết quả rõ ràng
    if len(bottom_text) >= 5:
        formatted = f"{top_text} {bottom_text[:3]} {bottom_text[3:]}"
    else:
        formatted = f"{top_text} {bottom_text}"

    # In kết quả
    print(f"📌 Biển số {i+1}: {formatted}")

    # Lưu ảnh biển số
    output_path = os.path.join(output_crop_path, f"{img_name}_plate_{i+1}.jpg")
    cv2.imwrite(output_path, plate_crop)
