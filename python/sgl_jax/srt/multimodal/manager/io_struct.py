from pydantic import BaseModel


class ImageGenerationsRequest(BaseModel):
    prompt: str
    model: str | None = None
    ## 生成的数量，影响latent的大小，需要padding
    n: int | None = 1
    response_format: str | None = "url"  # url | b64_json
    # 主要是weight和hight，也影响latent的大小，需要是可枚举的
    # 默认是 1280*720
    size: str | None = "1024x1024"  # e.g., 1024x1024
    ## 影响文本的后缀
    background: str | None = "auto"  # transparent | opaque | auto
    output_format: str | None = None  # png | jpeg | webp


class ImageResponse(BaseModel):
    id: str
    b64_json: str | None = None
    url: str | None = None


class VideoGenerationsRequest(BaseModel):
    prompt: str
    # 本地或者远程连接
    input_reference: str | None = None
    model: str | None = None
    # seconds,fps和 num_frames有相同的语义
    # num_frames 影响模型的输入，估计需要padding
    seconds: int | None = 4
    size: str | None = "720x1280"
    fps: int | None = None
    num_frames: int | None = None


class VideoResponse(BaseModel):
    id: str
    path: str | None = None
