import uvicorn
from fastapi import FastAPI,Response
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ffmpy3 import FFmpeg
import os

inputs={'./demo.mp4': None}
# 配置生成指令
outputs={None:'-c:v libx264 -c:a aac -strict -2 -f segment -segment_time 10 -segment_list_entry_prefix http://127.0.0.1:5000/video/ -segment_list ./demo.m3u8 ./demo-%4d.ts'}
app = FastAPI()
origins = [
    "*"
]
# 配置允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/video/{fileName}")
async def video(response: Response,fileName:str):
    response.headers["Content-Type"] = "application/x-mpegURL"
    print("收到文件请求")
    return FileResponse(fileName, filename=fileName) 

# 处理视频文件 生成 m3u8
def chuliVideo(path):
    # 判断文件是否存在
    if not os.path.exists(path):
        print("文件不存在")
        return False#文件不存在
    # 提取文件名
    print("文件存在")
    ff = FFmpeg(inputs=inputs, outputs=outputs)
    ff.run()

if __name__ == '__main__':
    chuliVideo('demo.mp4')
    uvicorn.run(app=app, host="127.0.0.1", port=5000)
