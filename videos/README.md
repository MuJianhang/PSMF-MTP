# PyCharm local workflow video / PyCharm 本地流程视频

Status: not yet recorded or uploaded. This directory currently contains recording guidance.

状态：尚未录制或上传视频，当前目录提供录制说明。

## Suggested recording sequence / 建议录制顺序

| Time / 时间 | Content / 内容 |
|---|---|
| 0-20 s | Show the full local project and prepared example inputs. / 展示完整本地项目及准备好的示例输入。 |
| 20-40 s | Show the PyCharm interpreter, working directory, input paths, 100-epoch weights, and threshold. / 展示解释器、工作目录、输入路径、100轮权重及阈值。 |
| 40-80 s | Run the actual local PSMF-MTP inference entry and show progress. / 运行真实的本地 PSMF-MTP 推理入口并展示过程。 |
| 80-105 s | Inspect predicted labels, probabilities, and available evaluation results. / 查看预测标签、概率及已有的评估结果。 |
| 105-120 s | Show how to substitute compatible own-data inputs and where results are saved. / 展示如何替换兼容的自有数据及输出位置。 |

The video uses the complete local project, including the retained core implementation. It can show run configurations, console output, input files, and result tables without displaying retained source code.

视频使用包含保留核心实现的完整本地项目。可以展示运行配置、控制台输出、输入文件和结果表格，无需展示保留的源码。

The demonstration should load the trained checkpoint for prediction. The 100-epoch training procedure can be explained in the README; the recording does not need to repeat training.

演示时加载已训练权重进行预测。100轮训练过程在 README 中说明，录屏中无需重新训练100轮。

After recording, upload the actual video to this directory and add its link to both READMEs. A suggested filename is `PSMF-MTP_PyCharm_demo.mp4`; no such video file is supplied yet.

录制完成后，将实际视频上传到本目录，并在中英文 README 中加入对应链接。建议文件名为 `PSMF-MTP_PyCharm_demo.mp4`；当前尚未提供该视频文件。
