# Face Window Tracker

这是一个用普通电脑摄像头驱动三维“虚拟窗口”的实验项目。

2026-09-06：已加入追踪稳定化、实测比例投影、可选 Spark 高斯渲染及回归测试。参见 [渲染与追踪改进记录](docs/rendering-tracking-review-2026-09-06.md)。

2026-09-07：新增可选的多点头部姿态追踪、相机内参配置、最新帧采集和渲染像素预算。参见 [本轮改动与验证](docs/cv-rendering-update-2026-09-07.md)。默认仍为轻量模式，增强模式使用 `face-tracker serve --tracker landmarker`。

摄像头只负责判断观看者的双眼位于屏幕的什么位置，不识别人是谁，也不保存人脸照片。网页根据观看位置实时重算透视投影：人向左移动时能看到更多物体右侧，人抬高时能看到更多物体顶部。屏幕边框始终不动，变化的是屏幕后方的观察视角。

隐私边界：应用自身不上传摄像头图像，但当前 MediaPipe 0.10.35 预编译包包含运行统计遥测，不能称为严格无外联。维护者表示不发送输入数据，且没有官方关闭开关；严格离线部署应另行验证源码构建或网络隔离。[维护者说明](https://github.com/google-ai-edge/mediapipe/issues/6291#issuecomment-4896121772)

仓库里已经包含两部分：

- Python 人脸位置服务：读取摄像头，检测人脸和双眼，输出经过平滑的位置数据。
- Three.js 展示箱：订阅位置数据，使用离轴投影渲染固定在屏幕后方的三维场景。

## 架构

项目没有把摄像头、视觉模型和三维渲染塞进同一个进程。运行时由两个独立进程组成：Python 服务独占摄像头并持续发布位置，浏览器只负责订阅数据和渲染。两边通过版本化 JSON 协议连接，后续替换视觉模型或渲染引擎时不必一起重写。

| 层 | 位置 | 主要职责 |
|---|---|---|
| 采集层 | `capture.py`、`service.py` | 独立取帧、单槽最新帧、断流重连及过期结果剔除 |
| 视觉层 | `tracker.py`、`landmark_tracker.py` | 轻量双眼检测 / 可选多点头部追踪 |
| 几何层 | `geometry.py`、`head_pose.py`、`calibration.py` | 眼间距估深 / PnP 姿态求解与相机内参 |
| 稳定层 | `filtering.py`、`stabilization.py` | One Euro Filter、位置关联与跳变拒绝 |
| 接口层 | `api.py` | 通过 REST 暴露状态，通过 WebSocket 推送实时结果 |
| 展示层 | `display-case.tsx` | 校准中心、映射坐标、计算离轴投影并渲染场景 |

### 数据流

数据从摄像头到画面的路径如下：

```text
电脑摄像头
  → OpenCV 持续取帧，仅保留最新一帧
  → 轻量 BlazeFace / 增强 Face Landmarker（二选一）
  → 眼间距估深 / 多点 PnP 求解观看者的 x / y / z 位置
  → One Euro Filter 抑制抖动
  → FastAPI WebSocket 推送位置
  → Three.js 离轴投影
  → 固定屏幕边框内的视角变化
```

### 人脸检测

默认使用 MediaPipe 的 BlazeFace Short Range Face Detector。它会返回人脸框和六个关键点，本项目只取左右眼关键点来计算眼睛中心和眼间距。

可选 `landmarker` 模式从面部关键点中选取 16 个参考点，用通用脸型和 OpenCV PnP 同时估计位置与朝向，针对“转头导致眼间距缩短，被误判为后退”的问题。它不是身份识别或视线追踪，通用脸型与假设瞳距仍会带来尺度偏差。两种模式不在运行中自动混合，以免不同距离模型造成跳变。

服务固定选择一张主要人脸。检测不到人脸时会发送 `tracking: false`，前端稍后把视角平滑复位。摄像头短暂断流时，服务会释放设备并自动重连。

### 三维位置估算

轻量模式中，双眼中点决定左右和上下位置，图像中的眼间距用于估算人与摄像头的距离。采用针孔相机模型：

```text
z = fx × IPD / eyeDistancePixels
x = (eyeCenterX - cx) × z / fx
y = -(eyeCenterY - cy) × z / fy
```

其中：

- `fx`、`fy` 默认根据摄像头水平视场角估算，也可载入实测内参；
- `cx`、`cy` 默认是图像中心，实测时使用标定主点；
- `IPD` 是假设的真实瞳距，默认 63 毫米；
- `eyeDistancePixels` 是两眼在画面中的像素距离。

这套方法能稳定判断相对移动，但目前还不是测量级定位。默认参数假设摄像头水平视场角为 70°，不同摄像头、不同瞳距都会带来距离比例误差。接口中的 `calibrated: false` 就是在明确标记这一点。

两种模式均支持 `FACE_CALIBRATION_PATH` 指向实测的 OpenCV 针孔内参 JSON。格式见 `config/camera.example.json`，该文件包含故意设置的示例保护标记，不能直接当成实测标定使用。内参标定不等于摄像头与屏幕外参、个人脸型和物理尺度全部标定；接口分别保留 `intrinsics_calibrated` 和 `calibrated`。

### 平滑处理

摄像头关键点会有小幅抖动。如果把原始坐标直接交给相机，三维场景会一直轻微晃动。后端分别对 `x`、`y`、`z` 使用 One Euro Filter：静止时加强平滑，快速移动时提高响应速度。

前端没有再使用固定帧数插值，而是按实际渲染时间计算插值比例。显示器刷新率变化或者识别帧率短时波动时，视角运动速度不会跟着改变。

### 离轴投影

展示箱没有旋转物体来假装视角变化，而是移动虚拟观察点并重算相机视锥。

箱口所在平面被视为真实屏幕平面。根据眼睛相对屏幕的位置，前端每帧计算投影矩阵的 `left`、`right`、`top` 和 `bottom`。因此：

- 屏幕四边与展示箱前框始终重合；
- 人移动时，前框保持固定；
- 箱内墙面、地面和展品产生不同程度的视差；
- 展品本身不转动，看到的是它在不同观察位置下的侧面。

前置摄像头的图像横坐标与观看者面对屏幕时的物理左右相反，前端已经对水平方向做了翻转。

## 运行环境

- macOS、Windows 或 Linux
- 可由 OpenCV 访问的摄像头
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Node.js 22.13 或更高版本
- npm

macOS 第一次启动摄像头时，需要允许终端或 Codex 使用摄像头。

## 启动项目

先安装 Python 依赖：

```bash
uv sync --extra dev
```

首次运行会从 MediaPipe 官方地址下载约 225 KB 的 BlazeFace 模型，保存在 `models/blaze_face_short_range.tflite`。

### 1. 启动人脸位置服务

推荐先使用 640 × 480。对于这项任务，更高的摄像头分辨率通常不会明显提高位置精度，却会增加图像传输和转换开销。

```bash
FACE_CAMERA_WIDTH=640 FACE_CAMERA_HEIGHT=480 uv run face-tracker serve
```

服务默认监听 `127.0.0.1:8765`。

Windows PowerShell 可在主项目目录使用增强模式（先停止占用同一摄像头的旧服务）：

```powershell
.\.venv\Scripts\face-tracker.exe serve --tracker landmarker
```

第一次会下载约 3.7 MB 的官方 Face Landmarker 模型。恢复轻量模式使用 `--tracker detector`。启动增强模式后，正对摄像头保持稳定约一秒完成中性位置校准，再尝试左右平移、轻微转头。

如果只想检查摄像头和检测结果，可以运行带标记的预览窗口：

```bash
FACE_CAMERA_WIDTH=640 FACE_CAMERA_HEIGHT=480 uv run face-tracker preview
```

在预览窗口中按 `Q` 或 `Esc` 退出。

### 2. 启动三维展示箱

打开另一个终端：

```bash
cd web
npm install
npm run dev
```

然后访问 <http://localhost:3000/>。

页面收集至少 10 个稳定有效样本后记录观看中心，不使用单帧校准；增强模式还要求初始姿态接近正面。右下角按钮依次用于切换人脸/鼠标控制、重新校准中心和进入全屏。

两个服务都可以用 `Ctrl+C` 停止。

## 本地接口

| 地址 | 用途 |
|---|---|
| `GET http://127.0.0.1:8765/api/v1/status` | 摄像头服务状态与错误信息 |
| `GET http://127.0.0.1:8765/api/v1/tracking/latest` | 最近一次追踪结果 |
| `WS ws://127.0.0.1:8765/ws/v1/tracking` | 实时位置数据 |
| `http://127.0.0.1:8765/docs` | FastAPI 生成的接口文档 |

WebSocket 只在产生新结果时发送数据，`sequence` 可用于判断是否收到重复帧。

结果增加 `tracker_backend`、`processing_ms`、`queue_age_ms`、`capture_dropped_frames`；增强模式增加 `calibration_ready`、`quality_reason` 和 `face.quality`。`frame.fps` 是取帧速率估计，不是网页帧率；`captured_at_unix_ms` 是读取完成时刻，不是传感器曝光时刻。缓存超过 500 ms 不再作为最新结果返回。

### 数据示例

```json
{
  "protocol_version": "1.0",
  "type": "face_tracking",
  "sequence": 317,
  "captured_at_unix_ms": 1788175503820,
  "frame": {
    "width": 640,
    "height": 480,
    "fps": 24.15
  },
  "tracking": true,
  "face": {
    "bbox": {
      "pixel": { "x": 341, "y": 100, "width": 203, "height": 203 },
      "normalized": {
        "x": 0.532813,
        "y": 0.208333,
        "width": 0.317188,
        "height": 0.422917
      }
    },
    "eyes": {
      "left": {
        "pixel": { "x": 513.469, "y": 173.567 },
        "screen_normalized": { "x": 0.60459, "y": 0.276806 }
      },
      "right": {
        "pixel": { "x": 440.199, "y": 166.846 },
        "screen_normalized": { "x": 0.375622, "y": 0.30481 }
      },
      "center": {
        "pixel": { "x": 476.834, "y": 170.206 },
        "screen_normalized": { "x": 0.490106, "y": 0.290808 }
      },
      "distance_pixels": 73.577
    },
    "viewer_position_m": {
      "raw": { "x": 0.134288, "y": 0.05976, "z": 0.391309 },
      "filtered": { "x": 0.135205, "y": 0.059625, "z": 0.392727 },
      "coordinate_system": "x-right_y-up_z-toward-viewer",
      "calibrated": false,
      "intrinsics_calibrated": false,
      "method": "eye-distance-assumed-ipd"
    },
    "head_rotation_deg": null,
    "facial_transformation_matrix": null
  }
}
```

`screen_normalized` 以画面中心为 `(0, 0)`，左下角为 `(-1, -1)`，右上角为 `(1, 1)`。

`viewer_position_m.filtered` 是展示端通常应当使用的位置。坐标系定义为：`x` 向摄像头画面右侧，`y` 向上，`z` 从摄像头指向观看者。

## 配置

所有后端参数都可以用环境变量覆盖。

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `FACE_TRACKER_BACKEND` | `detector` | `detector` 轻量 / `landmarker` 增强实验模式；命令行优先 |
| `FACE_LANDMARK_MODEL_PATH` | `models/face_landmarker.task` | 增强模式模型文件 |
| `FACE_CALIBRATION_PATH` | 空 | 实测针孔内参 JSON；同宽高比缩放，不自动补偿裁剪 |
| `FACE_CAMERA_SOURCE` | `0` | 摄像头编号，也可以是视频文件绝对路径 |
| `FACE_CAMERA_WIDTH` | `1280` | 请求的摄像头宽度 |
| `FACE_CAMERA_HEIGHT` | `720` | 请求的摄像头高度 |
| `FACE_CAMERA_FPS` | `30` | 请求帧率，最终结果取决于摄像头 |
| `FACE_CAMERA_HFOV_DEG` | `70` | 未标定时使用的水平视场角 |
| `FACE_ASSUMED_IPD_M` | `0.063` | 假设瞳距，单位为米 |
| `FACE_MODEL_PATH` | `models/blaze_face_short_range.tflite` | 模型文件位置 |
| `FACE_MODEL_URL` | MediaPipe 官方地址 | 模型不存在时的下载地址 |
| `FACE_MIN_DETECTION_CONFIDENCE` | `0.6` | 人脸检测置信度阈值 |
| `FACE_MIN_PRESENCE_CONFIDENCE` | `0.6` | 增强模式人脸存在阈值 |
| `FACE_MIN_TRACKING_CONFIDENCE` | `0.6` | 增强模式追踪阈值 |
| `FACE_FILTER_MIN_CUTOFF` | `1.2` | 静止时的平滑强度 |
| `FACE_FILTER_BETA` | `4.0` | 以米为输入的 One Euro 运动自适应系数；越大跟随越快 |
| `FACE_FILTER_DERIVATIVE_CUTOFF` | `1.0` | 速度估计的平滑强度 |

### 网页展示设置

页面右下角的齿轮按钮会打开**显示设置**。展示箱尺寸、观察投影与追踪映射、模型位置与材质、背景和灯光均可在浏览器中即时调整；这些设置只在当前页面会话内有效，刷新页面会恢复默认值。摄像头、检测器和后端滤波参数仍需通过上述 `FACE_*` 环境变量在启动服务前配置。

“外观与灯光”新增高斯滤波、覆盖范围和渲染像素预算。动态分辨率默认关闭；开启后只在持续卡顿时降低像素，稳定后缓慢恢复，不改变离轴投影。降低像素或缩小高斯覆盖范围会牺牲细节，不能替代更好的扫描模型。左上角分别显示渲染 FPS/P95 和有效 CV 样本速率。

模型的 `屏幕平面 Z` 以物理屏幕为 `0`：负值让模型位于屏幕内侧（展示箱深处），正值让它位于屏幕外侧、朝向观看者。它与“纵深跟随增益”不同；后者只影响虚拟观察点随人脸距离变化的幅度。

使用第二个摄像头：

```bash
FACE_CAMERA_SOURCE=1 uv run face-tracker preview
```

使用已有视频文件回放（按文件声明帧率节流；推理来不及时仍会跳帧，不是逐帧离线评测）：

```bash
FACE_CAMERA_SOURCE=/absolute/path/to/video.mp4 uv run face-tracker serve
```

## 项目结构

```text
.
├── src/face_tracker/
│   ├── api.py           # REST 与 WebSocket 接口
│   ├── cli.py           # serve / preview 命令
│   ├── config.py        # 环境变量与默认配置
│   ├── capture.py       # 最新帧采集与文件回放节流
│   ├── calibration.py   # 内参、畸变与缩放检查
│   ├── filtering.py     # One Euro Filter
│   ├── geometry.py      # 相机模型与三维位置估算
│   ├── head_pose.py     # 通用脸型 PnP 姿态求解
│   ├── landmark_tracker.py # 可选的多点视觉追踪
│   ├── tracker_factory.py  # 选择检测模式与模型
│   ├── service.py       # 摄像头生命周期、断流重连与数据缓存
│   └── tracker.py       # BlazeFace 推理与结果整理
├── tests/               # 后端测试
├── models/              # 首次运行后下载的模型
└── web/
    ├── app/             # 页面入口与全局样式
    ├── components/
    │   ├── display-case.tsx  # 展示箱、WebSocket 和离轴投影
    │   └── ui/button.tsx     # 页面使用的按钮组件
    └── public/          # 网站静态资源
```

Python 与 Three.js 之间只通过版本化 JSON 通信。以后如果换成 Unity、Unreal 或原生 OpenGL，后端不需要跟着重写；同样，替换检测算法时也可以保持 `v1` 协议不变。

## 性能说明

在 Apple M2、640 × 480 输入下，轻量检测器通常可以输出约 20–30 FPS。这个数字不是保证值：全屏 Retina WebGL、其他 Chromium 应用、视频播放和高负载的 `WindowServer` 都可能让帧率明显下降。

服务现已拆成持续取帧与只处理最新帧两个阶段，推理繁忙时覆盖未处理的旧帧。超过 250 ms 的输入或推理结果不发布，断流后清空缓存；这不能清除所有设备驱动内部的缓冲。带摄像头预览的 `preview` 命令仍是串行调试工具，不等同于服务端低延迟路径。

本机增强模式短时采样约 16 个结果/秒，推理约 9–11 ms，而取帧也约 16 Hz；当前样本没有显示推理积压。此处不是跨硬件保证，也不是端到端显示延迟。详细采样及失败情况见本轮记录。

前端渲染循环独立运行，识别帧率低于显示器刷新率时会在相邻位置之间平滑过渡。

## 已知限制

- 只追踪一张主要人脸，没有多人目标锁定。
- 不做身份识别、活体检测、视线追踪或表情分析。
- `z` 依赖假设瞳距、相机参数，增强模式还依赖通用脸型；尚未完成个人/屏幕联合实测标定。
- 普通单目摄像头无法在遮挡、强逆光或大角度侧脸下保证稳定检测。
- 当前展示端写死连接本机 `8765` 端口，适合本地原型，不适合直接部署到公网。
- 展示箱支持手动输入可见窗口物理宽度和中性观看距离，但摄像头相对屏幕的平移/倾角外参尚未建模。
- MediaPipe 预编译依赖的遥测未关闭，当前不承诺严格离线。

## 测试

后端测试：

```bash
uv run pytest
```

前端检查：

```bash
cd web
npm test
npm run typecheck
npm run lint
npm run build
```

`npm run build` 需要 Node.js 22.13 或更高版本。

GitHub 上的每次 `main` 推送和 Pull Request 都会运行同样的后端测试、前端检查与生产构建，配置位于 `.github/workflows/ci.yml`。

## 后续工作

如果要把原型做成稳定的展示设备，优先级比较高的工作是：

1. 标定摄像头内参、摄像头与屏幕的相对位置以及屏幕物理尺寸；
2. 在有实测参照的条件下对比两种 CV 模式的转头深度漂移、静止抖动和丢失恢复；
3. 基于真实序列调节质量门限与重新捕获，不盲目增大平滑或启用预测；
4. 用真实 GLB / glTF 展品替换当前程序化示例模型；
5. 建立可重复的精度、延迟和帧率测试流程。
