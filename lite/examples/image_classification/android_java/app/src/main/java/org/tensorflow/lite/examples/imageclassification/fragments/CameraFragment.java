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
package org.tensorflow.lite.examples.imageclassification.fragments;

import android.annotation.SuppressLint;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.LinearLayoutManager;
import com.google.common.util.concurrent.ListenableFuture;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.tensorflow.lite.examples.imageclassification.ImageClassifierHelper;
import org.tensorflow.lite.examples.imageclassification.R;
import org.tensorflow.lite.examples.imageclassification.databinding.FragmentCameraBinding;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.vision.classifier.Classifications;

/**
 * 相机Fragment，用于显示和控制设备摄像头、处理实时图像分类并展示结果。
 * 这是应用的核心UI界面。
 */
public class CameraFragment extends Fragment
        implements ImageClassifierHelper.ClassifierListener {
    private static final String TAG = "Image Classifier";

    private FragmentCameraBinding fragmentCameraBinding; // ViewBinding实例，用于访问布局中的视图
    private ImageClassifierHelper imageClassifierHelper; // 图像分类帮助类的实例
    private Bitmap bitmapBuffer; // 用于存储从CameraX获取的图像帧
    private ClassificationResultAdapter classificationResultsAdapter; // RecyclerView的适配器，用于显示分类结果
    private ImageAnalysis imageAnalyzer; // CameraX的图像分析用例
    private ProcessCameraProvider cameraProvider; // CameraX的相机提供者
    private final Object task = new Object();

    /**
     * 用于执行阻塞性相机操作的后台线程池。
     */
    private ExecutorService cameraExecutor;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater,
                             @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        // 使用ViewBinding加载布局
        fragmentCameraBinding =
                FragmentCameraBinding.inflate(inflater, container, false);
        return fragmentCameraBinding.getRoot();
    }

    @Override
    public void onResume() {
        super.onResume();
        // 检查是否已授予相机权限。如果未授予，则导航到权限请求Fragment。
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(
                    requireActivity(), R.id.fragment_container)
                    .navigate(
                            CameraFragmentDirections.actionCameraToPermissions());
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        // 关闭后台线程池，以防内存泄漏
        cameraExecutor.shutdown();
        // 同步清理分类器，释放TFLite相关资源
        synchronized (task) {
            imageClassifierHelper.clearImageClassifier();
        }
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        // 创建一个单线程的执行器来处理相机相关的后台任务
        cameraExecutor = Executors.newSingleThreadExecutor();
        // 初始化ImageClassifierHelper，并传入当前Fragment作为回调监听器
        imageClassifierHelper = ImageClassifierHelper.create(requireContext(), this);

        // 初始化用于显示结果的RecyclerView及其适配器
        classificationResultsAdapter = new ClassificationResultAdapter();
        classificationResultsAdapter.updateAdapterSize(imageClassifierHelper.getMaxResults());
        fragmentCameraBinding.recyclerviewResults.setAdapter(classificationResultsAdapter);
        fragmentCameraBinding.recyclerviewResults.setLayoutManager(new LinearLayoutManager(requireContext()));

        // 等待viewFinder（用于相机预览的View）加载完成后，再设置相机
        fragmentCameraBinding.viewFinder.post(this::setUpCamera);

        // 为底部控制面板（BottomSheet）中的控件附加监听器
        initBottomSheetControls();
    }

    /**
     * 当设备配置发生变化（如屏幕旋转）时调用。
     * 更新图像分析器的目标旋转角度。
     */
    @Override
    public void onConfigurationChanged(@NonNull Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
        if (imageAnalyzer != null) {
            imageAnalyzer.setTargetRotation(
                    fragmentCameraBinding.viewFinder.getDisplay().getRotation());
        }
    }

    // 初始化底部控制面板中的UI控件和监听器
    private void initBottomSheetControls() {
        // ... (省略UI控件的点击事件监听器设置，主要用于调整阈值、最大结果数、线程数等)
    }

    // 更新底部面板中显示的参数值，并重置分类器以应用新设置
    private void updateControlsUi() {
        fragmentCameraBinding.bottomSheetLayout.maxResultsValue.setText(
                String.valueOf(imageClassifierHelper.getMaxResults()));
        fragmentCameraBinding.bottomSheetLayout.thresholdValue.setText(
                String.format(Locale.US, "%.2f", imageClassifierHelper.getThreshold()));
        fragmentCameraBinding.bottomSheetLayout.threadsValue.setText(
                String.valueOf(imageClassifierHelper.getNumThreads()));
        
        // 清除当前的分类器实例。这会强制在下一次classify调用时，使用新的参数重新创建一个分类器。
        // 这样做是必要的，因为某些代理（如GPU代理）需要在其将被使用的线程上进行初始化。
        synchronized (task) {
            imageClassifierHelper.clearImageClassifier();
        }
    }

    /**
     * 初始化CameraX，并准备绑定相机用例。
     */
    private void setUpCamera() {
        // 获取CameraProvider的Future实例
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(requireContext());
        // 添加监听器，当CameraProvider可用时执行
        cameraProviderFuture.addListener(() -> {
            try {
                // 获取CameraProvider
                cameraProvider = cameraProviderFuture.get();
                // 绑定相机用例
                bindCameraUseCases();
            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "Camera setup failed", e);
            }
        }, ContextCompat.getMainExecutor(requireContext()));
    }

    /**
     * 声明并绑定预览、图像分析等用例到相机的生命周期。
     */
    @SuppressLint("UnsafeOptInUsageError")
    private void bindCameraUseCases() {
        if (cameraProvider == null) {
            throw new IllegalStateException("Camera initialization failed.");
        }

        // 1. 创建相机选择器，这里假设我们只使用后置摄像头
        CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();

        // 2. 创建预览(Preview)用例，并将其Surface提供给viewFinder
        Preview preview = new Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.getDisplay().getRotation())
                .build();
        preview.setSurfaceProvider(fragmentCameraBinding.viewFinder.getSurfaceProvider());

        // 3. 创建图像分析(ImageAnalysis)用例，这是执行实时推理的核心
        imageAnalyzer = new ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.getDisplay().getRotation())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST) // 只处理最新的图像帧，旧的直接丢弃
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888) // 设置输出格式为RGBA_8888
                .build();

        // 核心逻辑：为ImageAnalysis用例设置分析器(Analyzer)
        imageAnalyzer.setAnalyzer(cameraExecutor, image -> {
            if (bitmapBuffer == null) {
                // 为bitmapBuffer创建与图像帧匹配的Bitmap。这只在第一次时执行。
                bitmapBuffer = Bitmap.createBitmap(
                        image.getWidth(),
                        image.getHeight(),
                        Bitmap.Config.ARGB_8888);
            }
            // 将ImageProxy中的图像数据复制到我们的bitmapBuffer中
            image.getPlanes()[0].getBuffer().rewind();
            bitmapBuffer.copyPixelsFromBuffer(image.getPlanes()[0].getBuffer());

            // 获取图像的旋转角度
            int rotation = image.getImageInfo().getRotationDegrees();
            
            // 关闭ImageProxy，释放资源。非常重要！
            image.close();

            // 将准备好的Bitmap和旋转角度传递给ImageClassifierHelper进行分类
            // 推理将在ImageClassifierHelper内部的线程中执行
            synchronized (task) {
                if (imageClassifierHelper != null) {
                    imageClassifierHelper.classify(bitmapBuffer, rotation);
                }
            }
        });

        // 在重新绑定用例之前，必须先解绑所有旧的用例
        cameraProvider.unbindAll();

        try {
            // 将选择的相机和用例绑定到当前Fragment的生命周期
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer);
        } catch (Exception e) {
            Log.e(TAG, "Use case binding failed", e);
        }
    }

    // -- ImageClassifierHelper.ClassifierListener 的实现 --

    @Override
    public void onError(String error) {
        // 当分类发生错误时，在主线程中显示一个Toast提示
        requireActivity().runOnUiThread(() -> {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show();
            classificationResultsAdapter.updateResults(new ArrayList<>());
        });
    }

    @Override
    public void onResults(List<Classifications> results, long inferenceTime) {
        // 当收到分类结果时，在主线程中更新UI
        requireActivity().runOnUiThread(() -> {
            if (fragmentCameraBinding != null && fragmentCameraBinding.bottomSheetLayout != null) {
                 // 更新底部面板显示的推理耗时
                fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.setText(
                        String.format(Locale.US, "%d ms", inferenceTime));
            }
            
            // 将分类结果更新到RecyclerView的适配器中
            if (results != null && !results.isEmpty() && !results.get(0).getCategories().isEmpty()) {
                List<Category> sortedCategories = new ArrayList<>(results.get(0).getCategories());
                // 按置信度降序排序
                sortedCategories.sort((o1, o2) -> (int) (o2.getScore() * 100 - o1.getScore() * 100));
                classificationResultsAdapter.updateResults(sortedCategories);
            }
        });
    }
}