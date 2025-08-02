import logging


# 创建 logger
logger = logging.getLogger("SerialReceiver")
logger.setLevel(logging.DEBUG)  # 捕获所有级别的日志

# 创建控制台 handler（显示所有级别）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

# 创建文件 handler（仅记录 INFO 级别）
file_handler = logging.FileHandler("Run.log")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 添加 handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)
