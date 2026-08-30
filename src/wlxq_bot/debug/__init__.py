"""Debug Recorder：订阅任务执行事件，保存调试证据。

通过统一事件（DebugEvent）记录执行过程：
- 原始截图
- 标注截图
- 识别结果
- 动作日志

约束：Debug Recorder 只记录，不参与业务决策。
"""
