from paddlenlp import Taskflow
import time
import paddlehub as hub

chat_model = hub.Module(name="plato-mini")
dialogue = Taskflow("dialogue")


# warm up 一下
res = chat_model.predict(["你好"])

# 再计时
t1 = time.time()
res = chat_model.predict(["你好"])
t2 = time.time()
print(res)
print(f" chat_model 耗时：{t2-t1}s")



# warm UP 一下
res = dialogue(["你好"])

# 再计时
t1 = time.time()
res = dialogue(["你好"])
t2 = time.time()
print(res)
print(f" dialogue 耗时：{t2-t1}s")