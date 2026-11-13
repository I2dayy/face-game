# face-game

用摄像头玩的小游戏：移动头部，用嘴巴接住掉落的兔耳。

- 绿框（白色背景版）+1 分，红框（黑背景版）-1 分
- 重叠面积 ≥ 50% 算吃到；到 ±5 分会有飘字提示
- +10 获胜，-10 失败，按 q 退出

## 运行

pip install opencv-python numpy pillow
python Game_Pro.py

需要摄像头。图片资源放在同目录：rabbit_ears.png 为黑底原图，
rabbit_ears_clean.png / rabbit_ears_dirty.png 缺失时会自动从原图生成。
