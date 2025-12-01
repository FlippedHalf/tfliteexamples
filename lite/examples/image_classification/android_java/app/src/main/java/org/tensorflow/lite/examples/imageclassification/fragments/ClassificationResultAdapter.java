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
import android.view.LayoutInflater;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import org.tensorflow.lite.examples.imageclassification.databinding.ItemClassificationResultBinding;
import org.tensorflow.lite.support.label.Category;

/**
 * 用于在 RecyclerView 中显示图像分类结果列表的适配器。
 */
public class ClassificationResultAdapter
        extends RecyclerView.Adapter<ClassificationResultAdapter.ViewHolder> {
    // 当没有有效值时显示的占位符
    private static final String NO_VALUE = "--";
    // 存储分类结果的列表。这里使用Category，它包含了标签(label)和分数(score)。
    private List<Category> categories = new ArrayList<>();
    // 适配器期望显示的项目数量，即使没有足够的结果，也会显示占位符
    private int adapterSize = 0;

    /**
     * 更新分类结果列表，并刷新RecyclerView。
     * @param categories 新的分类结果列表
     */
    @SuppressLint("NotifyDataSetChanged")
    public void updateResults(List<Category> categories) {
        this.categories = new ArrayList<>(categories);
        // 通知RecyclerView整个数据集已更改，需要重绘
        notifyDataSetChanged();
    }

    /**
     * 更新适配器应显示的大小。这用于创建占位符项。
     * @param size 期望的列表大小
     */
    public void updateAdapterSize(int size) {
        adapterSize = size;
    }

    /**
     * 当 RecyclerView 需要一个新的 ViewHolder 时调用。
     * @param parent ViewHolder将要被添加到其中的父ViewGroup
     * @param viewType 视图类型
     * @return 一个新的 ViewHolder 实例
     */
    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        // 使用ViewBinding加载列表项的布局
        ItemClassificationResultBinding binding = ItemClassificationResultBinding
                .inflate(LayoutInflater.from(parent.getContext()), parent, false);
        return new ViewHolder(binding);
    }

    /**
     * 当 RecyclerView 需要将数据绑定到 ViewHolder 时调用。
     * @param holder 需要绑定数据的 ViewHolder
     * @param position 数据在列表中的位置
     */
    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        // 如果当前位置有数据，则绑定数据；否则显示占位符
        if (position < categories.size()) {
            holder.bind(categories.get(position));
        } else {
            holder.bind(null);
        }
    }

    /**
     * 返回列表中的项目总数。
     * @return 适配器应显示的项目总数
     */
    @Override
    public int getItemCount() {
        return adapterSize;
    }

    /**
     * ViewHolder 类，用于缓存列表项的视图，避免重复调用 findViewById。
     */
    public static class ViewHolder extends RecyclerView.ViewHolder {
        private final ItemClassificationResultBinding binding;

        public ViewHolder(@NonNull ItemClassificationResultBinding binding) {
            super(binding.getRoot());
            this.binding = binding;
        }

        /**
         * 将 Category 数据绑定到视图上。
         * @param category 要显示的数据。如果为null，则显示占位符。
         */
        public void bind(Category category) {
            if (category != null) {
                binding.tvLabel.setText(category.getLabel());
                binding.tvScore.setText(String.format(Locale.US, "%.2f", category.getScore()));
            } else {
                binding.tvLabel.setText(NO_VALUE);
                binding.tvScore.setText(NO_VALUE);
            }
        }
    }
}