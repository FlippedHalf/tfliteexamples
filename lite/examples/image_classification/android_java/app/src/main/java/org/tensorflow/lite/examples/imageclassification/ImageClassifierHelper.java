/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imageclassification;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import java.io.IOException;
import java.util.List;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;

/**
 * 图像分类器帮助类
 * 封装了所有与 TensorFlow Lite 图像分类相关的核心逻辑，包括：
 * - TFLite模型的加载和配置 (模型选择、硬件代理选择等)
 * - 图像的预处理
 * - 执行推理
 * - 将结果通过回调返回给UI层
 */
public class ImageClassifierHelper {
    private static final String TAG = "ImageClassifierHelper";
    // 定义常量，用于标识不同的推理代理 (Delegate)
    private static final int DELEGATE_CPU = 0;
    private static final int DELEGATE_GPU = 1;
    private static final int DELEGATE_NNAPI = 2;
    // 定义常量，用于标识不同的模型
    private static final int MODEL_MOBILENETV1 = 0;
    private static final int MODEL_EFFICIENTNETV0 = 1;
    private static final int MODEL_EFFICIENTNETV1 = 2;
    private static final int MODEL_EFFICIENTNETV2 = 3;

    // 推理相关的可调参数
    private float threshold;          // 分类置信度阈值，低于此值的结果将被过滤
    private int numThreads;           // 用于推理的线程数
    private int maxResults;           // 最大返回结果数
    private int currentDelegate;      // 当前使用的推理代理
    private int currentModel;         // 当前使用的模型

    private final Context context;                     // Android 应用上下文
    private final ClassifierListener imageClassifierListener; // 结果回调监听器
    private ImageClassifier imageClassifier;           // TFLite 图像分类器实例

    /**
     * 构造函数，初始化分类器帮助类
     * @param threshold         分类阈值
     * @param numThreads        线程数
     * @param maxResults        最大结果数
     * @param currentDelegate   当前推理代理
     * @param currentModel      当前模型
     * @param context           应用上下文
     * @param imageClassifierListener 回调监听器
     */
    public ImageClassifierHelper(Float threshold,
                                 int numThreads,
                                 int maxResults,
                                 int currentDelegate,
                                 int currentModel,
                                 Context context,
                                 ClassifierListener imageClassifierListener) {
        this.threshold = threshold;
        this.numThreads = numThreads;
        this.maxResults = maxResults;
        this.currentDelegate = currentDelegate;
        this.currentModel = currentModel;
        this.context = context;
        this.imageClassifierListener = imageClassifierListener;
        // 初始化时直接设置并创建分类器
        setupImageClassifier();
    }

    /**
     * 静态工厂方法，用于创建具有默认参数的 ImageClassifierHelper 实例。
     */
    public static ImageClassifierHelper create(
            Context context,
            ClassifierListener listener
    ) {
        return new ImageClassifierHelper(
                0.5f, // 默认阈值
                2,    // 默认线程数
                3,    // 默认最大结果数
                DELEGATE_CPU, // 默认使用 CPU
                MODEL_MOBILENETV1, // 默认使用 MobileNetV1
                context,
                listener
        );
    }

    /**
     * 设置并初始化 TFLite ImageClassifier。
     * 此方法会根据当前设置（模型、代理、线程数等）构建并创建分类器实例。
     */
    private void setupImageClassifier() {
        // 1. 设置分类器选项，如分数阈值和最大结果数
        ImageClassifier.ImageClassifierOptions.Builder optionsBuilder =
                ImageClassifier.ImageClassifierOptions.builder()
                        .setScoreThreshold(threshold)
                        .setMaxResults(maxResults);

        // 2. 设置通用的基础选项，如线程数和硬件代理
        BaseOptions.Builder baseOptionsBuilder =
                BaseOptions.builder().setNumThreads(numThreads);

        // 3. 根据用户选择配置硬件加速代理
        switch (currentDelegate) {
            case DELEGATE_CPU:
                // 默认使用CPU，无需特殊配置
                break;
            case DELEGATE_GPU:
                // 如果设备支持GPU代理，则启用GPU
                if (new CompatibilityList().isDelegateSupportedOnThisDevice()) {
                    baseOptionsBuilder.useGpu();
                } else {
                    imageClassifierListener.onError("GPU is not supported on this device");
                }
                break;
            case DELEGATE_NNAPI:
                // 启用 NNAPI 代理
                baseOptionsBuilder.useNnapi();
                break;
        }

        // 4. 根据用户选择确定要加载的模型文件名
        String modelName;
        switch (currentModel) {
            case MODEL_MOBILENETV1:
                modelName = "mobilenetv1.tflite";
                break;
            case MODEL_EFFICIENTNETV0:
                modelName = "efficientnet-lite0.tflite";
                break;
            case MODEL_EFFICIENTNETV1:
                modelName = "efficientnet-lite1.tflite";
                break;
            case MODEL_EFFICIENTNETV2:
                modelName = "efficientnet-lite2.tflite";
                break;
            default:
                modelName = "mobilenetv1.tflite";
                break;
        }

        // 5. 使用模型文件和配置选项创建 ImageClassifier 实例
        try {
            // Task Library 的核心方法，从 assets 加载模型并创建分类器
            imageClassifier = ImageClassifier.createFromFileAndOptions(context, modelName, optionsBuilder.build());
        } catch (IOException e) {
            imageClassifierListener.onError("Image classifier failed to initialize. See error logs for details");
            Log.e(TAG, "TFLite failed to load model with error: " + e.getMessage());
        }
    }

    /**
     * 对给定的 Bitmap 图像执行分类。
     * @param image 待分类的图像
     * @param imageRotation 图像的旋转角度，用于预处理时校正方向
     */
    public void classify(Bitmap image, int imageRotation) {
        // 如果分类器未初始化（例如，在切换模型或代理后），则重新设置
        if (imageClassifier == null) {
            setupImageClassifier();
        }

        // 记录推理开始时间
        long inferenceTime = SystemClock.uptimeMillis();

        // 1. 创建图像预处理器 (ImageProcessor)
        //    - 主要目的是根据摄像头传感器的方向，对图像进行旋转，以确保模型接收到的是正向图像
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder().add(new Rot90Op(-imageRotation / 90)).build();

        // 2. 预处理图像并转换为 TensorImage
        //    - TensorImage 是 TFLite 支持库中用于处理图像的格式
        TensorImage tensorImage = imageProcessor.process(TensorImage.fromBitmap(image));

        // 3. 执行分类
        //    - 这是核心推理步骤，模型将对输入的 tensorImage 进行分析
        List<Classifications> result = imageClassifier.classify(tensorImage);

        // 计算总推理耗时
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime;

        // 4. 通过回调将结果和耗时返回给调用方 (CameraFragment)
        imageClassifierListener.onResults(result, inferenceTime);
    }

    /**
     * 清除分类器实例。
     * 在切换模型或代理时调用，以释放资源并强制在下次分类时重建分类器。
     */
    public void clearImageClassifier() {
        imageClassifier = null;
    }

    // Getter 和 Setter 方法
    public float getThreshold() { return threshold; }
    public void setThreshold(float threshold) { this.threshold = threshold; }
    public int getNumThreads() { return numThreads; }
    public void setNumThreads(int numThreads) { this.numThreads = numThreads; }
    public int getMaxResults() { return maxResults; }
    public void setMaxResults(int maxResults) { this.maxResults = maxResults; }
    public void setCurrentDelegate(int currentDelegate) { this.currentDelegate = currentDelegate; }
    public void setCurrentModel(int currentModel) { this.currentModel = currentModel; }

    /**
     * 回调接口，用于将分类结果或错误信息异步传递回 UI 层。
     */
    public interface ClassifierListener {
        /**
         * 当发生错误时调用
         * @param error 错误信息
         */
        void onError(String error);

        /**
         * 当成功获得分类结果时调用
         * @param results 分类结果列表
         * @param inferenceTime 本次推理的耗时（毫秒）
         */
        void onResults(List<Classifications> results, long inferenceTime);
    }
}