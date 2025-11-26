# 项目介绍：TFLite 图像分类 Android 示例

## 1. 项目基本介绍

本项目是一个基于 TensorFlow Lite (TFLite) 的 Android 图像分类应用程序。它演示了如何使用 TFLite 在 Android 设备上对图像进行实时分类。

该应用程序的主要功能包括：

*   从设备的摄像头实时捕获视频流。
*   使用预训练的 TFLite 模型对捕获的图像进行分类。
*   在界面上显示分类结果和置信度。
*   允许用户切换不同的模型（如 MobileNet, EfficientNet）和推理设备（CPU, GPU, NNAPI）。

## 2. 项目整体架构

### 2.1 技术分层架构

项目的架构可以清晰地分为三个主要层次，自上而下依次是：

1.  **用户界面 (UI) 层**
    *   **职责**: 负责展示摄像头预览、用户交互以及显示最终的分类结果。
    *   **关键组件**:
        *   `CameraFragment`: 管理摄像头预览和图像捕获。
        *   `ClassificationResultAdapter`: 将分类结果展示在列表中。
    *   *数据流: UI 层将摄像头捕获的图像帧传递给应用逻辑层。*

2.  **应用逻辑层**
    *   **职责**: 作为 UI 层和底层推理层之间的桥梁，协调数据处理和业务逻辑。
    *   **关键组件**:
        *   `MainActivity`: 应用主窗口，管理各个 Fragment。
        *   `ImageClassifierHelper`: 封装所有 TFLite 相关的准备工作和调用，是本层的核心。
    *   *数据流: 应用逻辑层接收来自 UI 层的图像，调用推理层进行处理，然后将结果返回给 UI 层。*

3.  **TensorFlow Lite 推理层**
    *   **职责**: 执行实际的机器学习模型推理。
    *   **关键组件**:
        *   `TFLite Task Library`: 高级库，简化了图像分类任务的调用接口。
        *   `TFLite Model` (`.tflite` 文件): 预先训练好的、包含分类算法的神经网络模型。
    *   *数据流: 推理层接收来自应用逻辑层的预处理图像，执行模型，并返回原始分类结果。*

### 2.2 关键执行逻辑流程

核心的实时图像分类流程如下：

1.  **启动与初始化**:
    *   用户启动应用，`MainActivity` 加载 `CameraFragment`。
    *   `CameraFragment` 请求相机权限，并初始化 `CameraX` 来准备相机预览。
    *   同时，`CameraFragment` 创建一个 `ImageClassifierHelper` 的实例，这会触发 TFLite 模型的加载和初始化。

2.  **图像捕获 (循环执行)**:
    *   `CameraFragment` 通过 `ImageAnalysis` 用例从相机持续获取图像帧。

3.  **发起分类请求 (循环执行)**:
    *   对于每一帧图像，`CameraFragment` 调用 `imageClassifierHelper.classify()` 方法，并将图像数据（`Bitmap`）传递过去。

4.  **模型推理 (循环执行)**:
    *   `ImageClassifierHelper` 接收到图像后，首先进行预处理（如旋转图像以校正方向）。
    *   接着，它调用 `TFLite Task Library` 的接口，将预处理好的图像送入模型进行推理。

5.  **返回与显示结果 (循环执行)**:
    *   `TFLite Task Library` 推理完成后，将结果（分类列表和置信度）返回给 `ImageClassifierHelper`。
    *   `ImageClassifierHelper` 通过 `onResults()` 回调方法，将结果异步传递回 `CameraFragment`。
    *   `CameraFragment` 收到结果后，更新界面上的文本，向用户实时展示分类结果。

## 3. 项目关键文件介绍

### 3.1 `src` 目录下的关键文件

#### `ImageClassifierHelper.java`

这是项目的核心类，它封装了所有与 TensorFlow Lite 交互的逻辑。

*   **功能**:
    *   根据用户选择（模型、代理、线程数等）来配置和初始化 TFLite 的 `ImageClassifier`。
    *   提供 `classify()` 方法来对输入的 `Bitmap` 图像进行预处理和分类。
    *   通过 `ClassifierListener` 接口将分类结果异步传递回 UI 层。
*   **关键方法**:
    *   `setupImageClassifier()`:
        *   根据 `currentModel` (e.g., `MODEL_MOBILENETV1`) 确定要加载的 `.tflite` 文件名。
        *   根据 `currentDelegate` (CPU, GPU, or NNAPI) 配置推理硬件加速。
        *   使用 `ImageClassifier.createFromFileAndOptions()` 创建分类器实例。
    *   `classify(Bitmap image, int imageRotation)`:
        *   使用 `ImageProcessor` 对输入的图像进行旋转，以匹配模型训练时的方向。
        *   将 `Bitmap` 转换为 `TensorImage`。
        *   调用 `imageClassifier.classify(tensorImage)` 执行推理。
        *   通过 `onResults()` 回调返回结果。

#### `CameraFragment.java`

这个 Fragment 负责处理所有与相机相关的操作和 UI 显示。

*   **功能**:
    *   使用 CameraX `Preview` 和 `ImageAnalysis` 用例。
    *   将 `ImageAnalysis` 提供每一帧图像传递给 `ImageClassifierHelper`。
    *   实现了 `ImageClassifierHelper.ClassifierListener` 接口，用于接收和显示分类结果。
*   **关键逻辑**:
    *   在 `onViewCreated()` 中，它会检查相机权限。
    *   `bindCameraUseCases()`: 这是 CameraX 的核心设置部分，它将生命周期、`SurfaceProvider` (用于预览) 和 `ImageAnalysis.Analyzer` 绑定在一起。
    *   在 `ImageAnalysis.Analyzer` 的 `analyze()` 方法中，它获取最新的图像帧，并将其传递给 `imageClassifierHelper.classify()`。

### 3.2 配置文件

*   **`build.gradle` (app-level)**:
    *   定义了应用的 `applicationId`、`minSdk`、`targetSdk` 等。
    *   在 `dependencies` 代码块中，引入了 TensorFlow Lite Task Vision Library (`org.tensorflow:tensorflow-lite-task-vision`) 和 GPU 代理 (`org.tensorflow:tensorflow-lite-gpu`)。
    *   `android.androidResources.noCompress 'tflite'`：防止 Android 构建工具压缩 `.tflite` 模型文件，这对于确保模型能被 TFLite 正确加载至关重要。
*   **`download_models.gradle`**:
    *   一个独立的 Gradle 脚本，用于在构建项目时自动从网络下载预训练的 TFLite 模型文件，并将它们放置在 `assets` 目录下。
