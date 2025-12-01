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

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;
import org.tensorflow.lite.examples.imageclassification.R;

/**
 * 一个专门用于处理运行时权限请求的Fragment。
 * 它会在应用启动时检查并请求相机权限。
 */
public class PermissionsFragment extends Fragment {

    private static final String[] PERMISSIONS_REQUIRED = new String[]{Manifest.permission.CAMERA};

    /**
     * 使用新的 Activity Result API 注册一个权限请求启动器。
     * 当请求结果返回时，这个lambda表达式会被调用。
     */
    private final ActivityResultLauncher<String[]> requestPermissionLauncher = registerForActivityResult(
            new ActivityResultContracts.RequestMultiplePermissions(),
            permissions -> {
                // 检查是否所有请求的权限都已被授予
                boolean allGranted = true;
                for (Boolean isGranted : permissions.values()) {
                    if (!isGranted) {
                        allGranted = false;
                        break;
                    }
                }

                if (allGranted) {
                    // 如果所有权限都被授予，显示一个提示并导航到相机界面
                    Toast.makeText(requireContext(), "Permission request granted", Toast.LENGTH_LONG).show();
                    navigateToCamera();
                } else {
                    // 如果有任何权限被拒绝，显示一个提示。
                    // 在实际应用中，这里可能需要向用户解释为什么需要这个权限，并提供一个重新请求的选项。
                    Toast.makeText(requireContext(), "Permission request denied", Toast.LENGTH_LONG).show();
                }
            });

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // 检查是否已经拥有所有必需的权限
        if (!hasPermissions(requireContext())) {
            // 如果没有，则启动权限请求流程
            requestPermissionLauncher.launch(PERMISSIONS_REQUIRED);
        } else {
            // 如果已经拥有权限，直接导航到相机界面
            navigateToCamera();
        }
    }

    /**
     * 导航到相机Fragment。
     */
    private void navigateToCamera() {
        // 使用 Navigation 组件安全地从当前Fragment跳转到CameraFragment
        // actionPermissionsToCamera 是在 nav_graph.xml 中定义的 action
        Navigation.findNavController(requireActivity(), R.id.fragment_container)
                .navigate(PermissionsFragmentDirections.actionPermissionsToCamera());
    }

    /**
     * 一个静态的辅助方法，用于检查应用是否已经获得了所有必需的权限。
     * @param context 应用上下文
     * @return 如果所有权限都已被授予，则返回true；否则返回false。
     */
    public static boolean hasPermissions(Context context) {
        for (String permission : PERMISSIONS_REQUIRED) {
            if (ContextCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }
}