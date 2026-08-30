"""Action Executor：在安全检查通过后执行输入并验证结果。

包含三个子模块：
- Executor：动作执行入口，串联 Safety Guard 和 Input Controller
- Input Controller：封装 PyAutoGUI 鼠标、键盘输入
- Safety Guard：停止信号、失败次数、窗口状态和动作边界检查

约束：任务代码只能通过 Action Executor 执行输入，
禁止绕过 Safety Guard 直接调用 Input Controller。
"""
