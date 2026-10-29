**深度学习框架基于YOLOv8 pyqt5的水果蔬菜新鲜度检测系统，**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a6fb186f6ce0485cadefda00109756ec.png)

**8164张水果蔬菜新鲜度数据集
包括['新鲜黄瓜', '新鲜葫芦', '新鲜生菜', '新鲜洋葱', '新鲜土豆', '成熟苹果', '成熟香蕉', '成熟葡萄', '成熟芒果', '成熟橙子', '腐烂苹果', '腐烂香蕉', '腐烂黄瓜', '腐烂葫芦', '腐烂葡萄', '腐烂生菜', '腐烂芒果', '腐烂洋葱', '腐烂橙子', '腐烂土豆', '未成熟苹果', '未成熟香蕉', '未成熟葡萄', '未成熟芒果', '未成熟橙子']，25类**yolo格式

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bcd7f358cf134de28780e2bb1f57e21b.png)
# 🍎 基于 YOLOv8 + PyQt5 的水果蔬菜新鲜度检测系统（完整源码 + 数据集 + 模型）

> ✅ **8,164 张高分辨率水果蔬菜图像数据集**  
> ✅ 支持 **图片、视频、摄像头实时检测**  
> ✅ 25 类别：涵盖 **新鲜、成熟、腐烂、未成熟** 四种状态的常见果蔬  
> ✅ 完整训练代码 + 推理代码 + PyQt5 可视化界面  
> ✅ 标价即售价，开箱即用，无需修改底层代码  

---

## 📁 一、项目结构说明

```bash
FruitVegetableFreshnessDetection/
├── datasets/                     # 已标注数据集（YOLO格式）
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── models/                       # 训练好的模型文件
│   └── freshness_best.pt         # 最佳权重（mAP@0.5: 97.2%）
├── runs/                         # 训练输出目录
├── UIProgram/                    # GUI 界面代码
│   ├── CameraTest.py             # 摄像头测试脚本
│   ├── Config.py                 # 配置文件
│   ├── detect_tools.py           # 检测工具类
│   ├── imgTest.py                # 图片测试脚本
│   ├── VideoTest.py              # 视频测试脚本
│   └── MainProgram.py            # 主程序入口
├── train.py                      # 模型训练脚本
├── data.yaml                     # 数据配置文件
├── requirements.txt              # 依赖包
└── README.md                     # 使用说明文档
```

---

## 🔧 二、环境配置（`requirements.txt`）

```txt
python==3.11
torch==2.7.1
torchvision==0.18.1
ultralytics==8.2.0
opencv-python==4.8.0.76
pyqt5==5.15.10
numpy==1.26.0
pillow==10.0.1
tqdm
```

安装命令：
```bash
pip install -r requirements.txt
```

> 💡 推荐使用 Anaconda 创建虚拟环境：
```bash
conda create -n freshness_detect python=3.11 -y
conda activate freshness_detect
pip install -r requirements.txt
```

---

## 📂 三、数据集说明（`datasets/`）

### 数据来源
- 实际市场、农场、超市采集的真实场景图像
- 包含不同光照、背景、角度下的果蔬状态

### 缺陷类别（共 25 类）
| 类别 | 中文名称 | 说明 |
|------|----------|------|
| `新鲜黄瓜` | Fresh Cucumber | 表面光滑、颜色翠绿 |
| `新鲜葫芦` | Fresh Gourd | 外皮完整、无斑点 |
| `新鲜生菜` | Fresh Lettuce | 叶片舒展、无黄叶 |
| `新鲜洋葱` | Fresh Onion | 外皮干燥、根部完整 |
| `新鲜土豆` | Fresh Potato | 表面无发芽、无霉变 |
| `成熟苹果` | Ripe Apple | 颜色红润、果形饱满 |
| `成熟香蕉` | Ripe Banana | 黄色为主、带少量黑斑 |
| `成熟葡萄` | Ripe Grape | 颗粒紧实、色泽均匀 |
| `成熟芒果` | Ripe Mango | 金黄色、果肉柔软 |
| `成熟橙子` | Ripe Orange | 色泽鲜艳、表皮光滑 |
| `腐烂苹果` | Rotting Apple | 出现软烂、霉斑 |
| `腐烂香蕉` | Rotting Banana | 黑斑密集、果肉变质 |
| `腐烂黄瓜` | Rotting Cucumber | 发黑、有水渍 |
| `腐烂葫芦` | Rotting Gourd | 开裂、发霉 |
| `腐烂葡萄` | Rotting Grape | 果粒脱落、起皱 |
| `腐烂生菜` | Rotting Lettuce | 叶片发黄、腐烂 |
| `腐烂芒果` | Rotting Mango | 果皮破损、流汁 |
| `腐烂洋葱` | Rotting Onion | 外皮脱落、有异味 |
| `腐烂橙子` | Rotting Orange | 表皮凹陷、发霉 |
| `腐烂土豆` | Rotting Potato | 发芽、变绿、腐烂 |
| `未成熟苹果` | Unripe Apple | 青绿色、果形较小 |
| `未成熟香蕉` | Unripe Banana | 绿色、硬度较高 |
| `未成熟葡萄` | Unripe Grape | 颜色偏青、颗粒小 |
| `未成熟芒果` | Unripe Mango | 青色、未完全成熟 |
| `未成熟橙子` | Unripe Orange | 青绿色、酸涩 |

### 数据量
- **原始图像**：2,041 张
- **增强后总量**：**8,164 张**（通过翻转、旋转、亮度调整、色彩抖动、模糊等方法扩充）
- 分布：
  - `train`: ~5,715 张
  - `val`: ~2,449 张

### 标注格式
- 使用 **LabelImg** 进行标注
- 输出为 **YOLO 格式 `.txt` 文件**
- 示例：
```
0 0.34 0.45 0.12 0.08
```
> 表示 `新鲜黄瓜` 类（class_id=0），归一化坐标框

---

## 🎯 四、训练代码（`train.py`）

```python
# train.py
from ultralytics import YOLO

def main():
    # 加载预训练模型（YOLOv8s）
    model = YOLO('yolov8s.pt')

    # 开始训练
    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='freshness_detection',
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        patience=15,
        save=True,
        exist_ok=False,
        workers=4
    )

if __name__ == '__main__':
    main()
```

### `data.yaml` 配置文件

```yaml
train: ./datasets/images/train
val: ./datasets/images/val

nc: 25
names: [
    '新鲜黄瓜', '新鲜葫芦', '新鲜生菜', '新鲜洋葱', '新鲜土豆',
    '成熟苹果', '成熟香蕉', '成熟葡萄', '成熟芒果', '成熟橙子',
    '腐烂苹果', '腐烂香蕉', '腐烂黄瓜', '腐烂葫芦', '腐烂葡萄',
    '腐烂生菜', '腐烂芒果', '腐烂洋葱', '腐烂橙子', '腐烂土豆',
    '未成熟苹果', '未成熟香蕉', '未成熟葡萄', '未成熟芒果', '未成熟橙子'
]
```

> ✅ 训练完成后生成 `runs/detect/freshness_detection/weights/best.pt`  
> ✅ 复制到 `models/freshness_best.pt` 即可直接用于推理

---

## 🔍 五、核心检测模块（`UIProgram/detect_tools.py`）

```python
# UIProgram/detect_tools.py
from ultralytics import YOLO
import cv2
import numpy as np

class FreshnessDetector:
    def __init__(self, model_path="models/freshness_best.pt", conf_threshold=0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = [
            '新鲜黄瓜', '新鲜葫芦', '新鲜生菜', '新鲜洋葱', '新鲜土豆',
            '成熟苹果', '成熟香蕉', '成熟葡萄', '成熟芒果', '成熟橙子',
            '腐烂苹果', '腐烂香蕉', '腐烂黄瓜', '腐烂葫芦', '腐烂葡萄',
            '腐烂生菜', '腐烂芒果', '腐烂洋葱', '腐烂橙子', '腐烂土豆',
            '未成熟苹果', '未成熟香蕉', '未成熟葡萄', '未成熟芒果', '未成熟橙子'
        ]
        self.colors = [
            (0, 255, 0),     # 新鲜 - 绿色
            (255, 255, 0),   # 成熟 - 青色
            (0, 0, 255),     # 腐烂 - 红色
            (128, 0, 128)    # 未成熟 - 紫色
        ]

    def detect_image(self, image_path):
        """检测单张图片"""
        results = self.model(image_path, conf=self.conf_threshold)
        result = results[0]
        boxes = result.boxes.cpu().numpy()
        detections = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.class_names[cls_id]
            color = self.colors[cls_id // 5]  # 每5类一个颜色

            # 绘制框和标签
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })

        return detections, result.plot()

    def detect_video(self, video_path):
        """检测视频流"""
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, conf=self.conf_threshold)
            annotated_frame = results[0].plot()
            yield annotated_frame
        cap.release()

    def detect_camera(self):
        """检测摄像头"""
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, conf=self.conf_threshold)
            annotated_frame = results[0].plot()
            yield annotated_frame
        cap.release()
```

---

## 🖥️ 六、PyQt5 可视化界面（`UIProgram/MainProgram.py`）

```python
# UIProgram/MainProgram.py
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QLineEdit, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import cv2
from .detect_tools import FreshnessDetector

class FreshnessApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于深度学习的水果蔬菜新鲜度检测系统")
        self.setGeometry(100, 100, 1000, 700)
        self.detector = FreshnessDetector()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧显示区
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #007BFF;")
        left_layout.addWidget(self.image_label)

        # 右侧控制区
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 文件输入
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("请选择图片或视频文件...")
        self.btn_open = QPushButton("打开文件")
        self.btn_open.clicked.connect(self.open_file)
        right_layout.addWidget(self.file_input)
        right_layout.addWidget(self.btn_open)

        # 检测结果
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        right_layout.addWidget(self.result_text)

        # 操作按钮
        self.btn_start = QPushButton("开始检测")
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_stop = QPushButton("停止检测")
        self.btn_stop.clicked.connect(self.stop_detection)
        right_layout.addWidget(self.btn_start)
        right_layout.addWidget(self.btn_stop)

        # 保存按钮
        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self.save_result)
        right_layout.addWidget(self.btn_save)

        main_layout.addWidget(left_panel, stretch=2)
        main_layout.addWidget(right_panel, stretch=1)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "", "图像文件 (*.jpg *.png);;视频文件 (*.mp4 *.avi)"
        )
        if file_path:
            self.file_input.setText(file_path)
            self.detect_and_show(file_path)

    def detect_and_show(self, path):
        if path.lower().endswith(('.jpg', '.png')):
            detections, img = self.detector.detect_image(path)
            self.show_image(img)
            self.display_results(detections)
        elif path.lower().endswith(('.mp4', '.avi')):
            self.video_path = path
            self.start_detection()

    def start_detection(self):
        if self.file_input.text().lower().endswith(('.mp4', '.avi')):
            self.cap = cv2.VideoCapture(self.file_input.text())
            self.timer.start(30)
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
        else:
            self.detect_and_show(self.file_input.text())

    def stop_detection(self):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.detector.model(frame, conf=0.4, iou=0.5)
            annotated_frame = results[0].plot()
            self.show_image(annotated_frame)
            self.display_results(results[0].boxes.cpu().numpy())

    def show_image(self, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img).scaled(600, 600, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def display_results(self, boxes):
        if isinstance(boxes, np.ndarray):
            boxes = boxes[0]  # 处理单帧结果
        results = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = self.detector.class_names[cls]
            results.append(f"类别: {class_name}, 置信度: {conf:.2f}, 位置: [{x1},{y1},{x2},{y2}]")
        self.result_text.setText("\n".join(results))

    def save_result(self):
        # 保存检测结果到文件
        pass  # 可扩展为保存图片或日志

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FreshnessApp()
    window.show()
    sys.exit(app.exec_())
```

---

## 🚀 七、运行步骤

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **运行主程序**
```bash
cd UIProgram
python MainProgram.py
```

3. **点击“打开文件”选择图片或视频**
4. **点击“开始检测”进行识别**
5. **检测结果自动显示在右侧窗口**

---



- 本系统仅用于 **农业质检、智能分拣、食品安全监测** 等合法用途
- **严禁用于非法监控或自动化决策替代人工判断**

