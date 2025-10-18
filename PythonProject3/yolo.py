from ultralytics import YOLO

# 加载YOLOv8预训练模型（首次运行会自动下载，约60MB）
model = YOLO("yolov8n.pt")  # "n"代表轻量版，速度快，适合入门

# 检测一张图片（可以替换成自己的图片路径，如"test.jpg"）
results = model("https://ultralytics.com/images/zidane.jpg")  # 用示例图片，无需自己准备

# 显示检测结果（会弹出窗口展示带框的图片）
results[0].show()
print(results[0].boxes)

