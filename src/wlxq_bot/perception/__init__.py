"""Perception Pipeline：把窗口截图转换为可供任务判断的界面状态。

包含三个子模块：
- Screen Capture：获取游戏窗口客户区截图
- Vision：模板匹配、颜色识别、调试标注
- Locator：坐标换算、ROI 计算和动作点定位

依赖方向：Task Engine -> Perception Pipeline，不反向依赖。
"""
