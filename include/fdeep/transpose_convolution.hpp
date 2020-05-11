// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/convolution.hpp"
#include "fdeep/common.hpp"

#include "fdeep/filter.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace fdeep { namespace internal
{

inline tensor convolve_im2col_transpose(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t strides_y,
    std::size_t strides_x,
    const im2col_filter_matrix& filter_mat,
    const tensor& in_padded)
{
    const auto fy = filter_mat.filter_shape_.height_;
    const auto fx = filter_mat.filter_shape_.width_;
    const auto fz = filter_mat.filter_shape_.depth_;
    ColMajorMatrixXf a(fy * fx * fz + 1, out_height * out_width);
    EigenIndex a_x = 0;
    for (std::size_t y = 0; y < out_height; ++y)
    {
        for (std::size_t x = 0; x < out_width; ++x)
        {
            EigenIndex a_y = 0;
            for (std::size_t yf = 0; yf < fy; ++yf)
            {
                for (std::size_t xf = 0; xf < fx; ++xf)
                {
                    for (std::size_t zf = 0; zf < fz; ++zf)
                    {
                        a(a_y++, a_x) = in_padded.get_ignore_rank(tensor_pos(
                                fplus::floor(static_cast<float>(y) / strides_y + 0.001) + yf,
                                fplus::floor(static_cast<float>(x) / strides_x + 0.001) + xf,
                                zf));
                    }
                }
                a(a_y, a_x) = static_cast<float_type>(1);
            }
            ++a_x;
        }
    }

    const std::size_t val_cnt =
        static_cast<std::size_t>(filter_mat.mat_.rows() * a.cols());
    assertion(val_cnt % (out_height * out_width) == 0,
        "Can not calculate out_depth");

    const std::size_t out_depth = val_cnt / (out_height * out_width);
    assertion(val_cnt == out_depth * out_height * out_width,
        "Invalid target size");

    shared_float_vec res_vec = fplus::make_shared_ref<float_vec>();
    res_vec->resize(static_cast<std::size_t>(out_depth * out_height * out_width));

    Eigen::Map<ColMajorMatrixXf, Eigen::Unaligned> out_mat_map(
        res_vec->data(),
        static_cast<EigenIndex>(filter_mat.mat_.rows()),
        static_cast<EigenIndex>(a.cols()));
    
    // https://en.wikipedia.org/wiki/Toeplitz_matrix
    out_mat_map.noalias() = filter_mat.mat_ * a;

    return tensor(
        tensor_shape_with_changed_rank(
            tensor_shape(out_height, out_width, out_depth),
            in_padded.shape().rank()),
        res_vec);
}

inline tensor convolve_vol2col_transpose(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t out_depth,
    std::size_t strides_y,
    std::size_t strides_x,
    std::size_t strides_z,
    const im2col_filter_matrix& filter_mat,
    const tensor& in_padded)
{
    const auto fdim4 = filter_mat.filter_shape_.size_dim_4_;
    const auto fy = filter_mat.filter_shape_.height_;
    const auto fx = filter_mat.filter_shape_.width_;
    const auto fz = filter_mat.filter_shape_.depth_;
    ColMajorMatrixXf a(fdim4 * fy * fx * fz + 1, out_height * out_width * out_depth);
    EigenIndex a_x = 0;
    for (std::size_t y = 0; y < out_height; ++y)
    {
        for (std::size_t x = 0; x < out_width; ++x)
        {
            for (std::size_t z = 0; z < out_depth; ++z)
            {
                EigenIndex a_y = 0;
                for (std::size_t yf = 0; yf < fdim4; ++yf)
                {
                    for (std::size_t xf = 0; xf < fy; ++xf)
                    {
                        for (std::size_t zf = 0; zf < fx; ++zf)
                        {
                            for (std::size_t c = 0; c < fz; ++c)
                            {
                                a(a_y++, a_x) = in_padded.get_ignore_rank(tensor_pos(
                                fplus::floor(static_cast<float>(y) / strides_y + 0.001) + yf,
                                fplus::floor(static_cast<float>(x) / strides_x + 0.001) + xf,
                                fplus::floor(static_cast<float>(z) / strides_z + 0.001) + zf,
                                        c));
                            }
                        }
                        a(a_y, a_x) = static_cast<float_type>(1);
                    }
                }
                ++a_x;
            }
        }
    }


    const std::size_t val_cnt =
    static_cast<std::size_t>(filter_mat.mat_.rows() * a.cols());
    assertion(val_cnt % (out_height * out_width * out_depth) == 0,
        "Can not calculate out_depth");

    const std::size_t out_filters = val_cnt / (out_height * out_width * out_depth);
    assertion(val_cnt == out_filters * out_height * out_width  * out_depth,
        "Invalid target size");

    shared_float_vec res_vec = fplus::make_shared_ref<float_vec>();
    res_vec->resize(static_cast<std::size_t>(out_filters * out_height * out_width * out_depth));

    Eigen::Map<ColMajorMatrixXf, Eigen::Unaligned> out_mat_map(
        res_vec->data(),
        static_cast<EigenIndex>(filter_mat.mat_.rows()),
        static_cast<EigenIndex>(a.cols()));
 
 // https://en.wikipedia.org/wiki/Toeplitz_matrix
    out_mat_map.noalias() = filter_mat.mat_ * a;

    return tensor(
        tensor_shape_with_changed_rank(
            tensor_shape(out_height, out_width, out_depth, out_filters),
            in_padded.shape().rank()),
        res_vec);
}

inline convolution_config preprocess_transpose_convolution(
    const shape2& filter_shape,
    const shape2& strides,
    padding pad_type,
    std::size_t input_shape_height,
    std::size_t input_shape_width)
{
    // https://www.tensorflow.org/api_guides/python/nn#Convolution
    const int filter_height = static_cast<int>(filter_shape.height_);
    const int filter_width = static_cast<int>(filter_shape.width_);
    const int in_height = static_cast<int>(input_shape_height);
    const int in_width = static_cast<int>(input_shape_width);
    const int strides_y = static_cast<int>(strides.height_);
    const int strides_x = static_cast<int>(strides.width_);

    int out_height = 0;
    int out_width = 0;

    if (pad_type == padding::same || pad_type == padding::causal)
    {
        out_height = in_height * strides_y;
        out_width  = in_width * strides_x;
    }
    else
    {
        out_height = strides_y * in_height + filter_height - strides_y;
        out_width = strides_x * in_width + filter_width - strides_x;
    }

    int pad_top = 0;
    int pad_bottom = 0;
    int pad_left = 0;
    int pad_right = 0;

    if (pad_type == padding::same)
    {
        int pad_along_height = 0;
        int pad_along_width = 0;

        if (in_height % strides_y == 0)
            pad_along_height = std::max(filter_height - strides_y, 0);
        else
            pad_along_height = std::max(filter_height - (in_height % strides_y), 0);
        if (in_width % strides_x == 0)
            pad_along_width = std::max(filter_width - strides_x, 0);
        else
            pad_along_width = std::max(filter_width - (in_width % strides_x), 0);

        pad_top = pad_along_height / 2;
        pad_bottom = pad_along_height - pad_top;
        pad_left = pad_along_width / 2;
        pad_right = pad_along_width - pad_left;
    }
    else if (pad_type == padding::causal)
    {
        pad_top = filter_height - 1;
        pad_left = filter_width - 1;
    }
    
    std::size_t out_height_size_t = fplus::integral_cast_throw<std::size_t>(out_height);
    std::size_t out_width_size_t = fplus::integral_cast_throw<std::size_t>(out_width);
    std::size_t pad_top_size_t = fplus::integral_cast_throw<std::size_t>(pad_top);
    std::size_t pad_bottom_size_t = fplus::integral_cast_throw<std::size_t>(pad_bottom);
    std::size_t pad_left_size_t = fplus::integral_cast_throw<std::size_t>(pad_left);
    std::size_t pad_right_size_t = fplus::integral_cast_throw<std::size_t>(pad_right);

//    std::cout << "Pad top: " << pad_top << ", pad left: " << pad_left << ", out_size: " << out_height << "x" << out_width << std::endl;
    return {pad_top_size_t, pad_bottom_size_t,
        pad_left_size_t, pad_right_size_t,
        out_height_size_t, out_width_size_t};
}

inline convolution_config3D preprocess_transpose_convolution3D(
    const shape3& filter_shape,
    const shape3& strides,
    padding pad_type,
    std::size_t input_shape_height,
    std::size_t input_shape_width,
    std::size_t input_shape_depth)
{
    // https://www.tensorflow.org/api_guides/python/nn#Convolution
    const int filter_height = static_cast<int>(filter_shape.height_);
    const int filter_width = static_cast<int>(filter_shape.width_);
    const int filter_depth = static_cast<int>(filter_shape.depth_);
    const int in_height = static_cast<int>(input_shape_height);
    const int in_width = static_cast<int>(input_shape_width);
    const int in_depth = static_cast<int>(input_shape_depth);
    const int strides_y = static_cast<int>(strides.height_);
    const int strides_x = static_cast<int>(strides.width_);
    const int strides_z = static_cast<int>(strides.depth_);

    int out_height = 0;
    int out_width = 0;
    int out_depth = 0;

    
    if (pad_type == padding::same || pad_type == padding::causal)
    {
        out_height = in_height * strides_y;
        out_width  = in_width * strides_x;
        out_depth  = in_depth * strides_z;
    }
    else
    {
        out_height = strides_y * in_height + filter_height - strides_y;
        out_width = strides_x * in_width + filter_width - strides_x;
        out_depth = strides_z * in_depth + filter_depth - strides_z;
    }

    int pad_top = 0;
    int pad_bottom = 0;
    int pad_left = 0;
    int pad_right = 0;
    int pad_front = 0;
    int pad_back = 0;

    if (pad_type == padding::same)
    {
        int pad_along_height = 0;
        int pad_along_width = 0;
        int pad_along_depth = 0;

        if (in_height % strides_y == 0)
            pad_along_height = std::max(filter_height - strides_y, 0);
        else
            pad_along_height = std::max(filter_height - (in_height % strides_y), 0);
        if (in_width % strides_x == 0)
            pad_along_width = std::max(filter_width - strides_x, 0);
        else
            pad_along_width = std::max(filter_width - (in_width % strides_x), 0);
        if (in_depth % strides_z == 0)
            pad_along_depth = std::max(filter_depth - strides_z, 0);
        else
            pad_along_depth = std::max(filter_depth - (in_depth % strides_z), 0);

        pad_top = pad_along_height / 2;
        pad_bottom = pad_along_height - pad_top;
        pad_left = pad_along_width / 2;
        pad_right = pad_along_width - pad_left;
        pad_front = pad_along_depth / 2;
        pad_back = pad_along_depth - pad_front;
    }
    else if (pad_type == padding::causal)
    {
        pad_top = filter_height - 1;
        pad_left = filter_width - 1;
        pad_front = filter_depth - 1;

    }

    std::size_t out_height_size_t = fplus::integral_cast_throw<std::size_t>(out_height);
    std::size_t out_width_size_t = fplus::integral_cast_throw<std::size_t>(out_width);
    std::size_t out_depth_size_t = fplus::integral_cast_throw<std::size_t>(out_depth);
    std::size_t pad_top_size_t = fplus::integral_cast_throw<std::size_t>(pad_top);
    std::size_t pad_bottom_size_t = fplus::integral_cast_throw<std::size_t>(pad_bottom);
    std::size_t pad_left_size_t = fplus::integral_cast_throw<std::size_t>(pad_left);
    std::size_t pad_right_size_t = fplus::integral_cast_throw<std::size_t>(pad_right);
    std::size_t pad_front_size_t = fplus::integral_cast_throw<std::size_t>(pad_front);
    std::size_t pad_back_size_t = fplus::integral_cast_throw<std::size_t>(pad_back);

    return {pad_top_size_t, pad_bottom_size_t,
        pad_left_size_t, pad_right_size_t,
        pad_front_size_t, pad_back_size_t,
        out_height_size_t, out_width_size_t,
        out_depth_size_t};
}

inline tensor transpose_convolve(
    const shape2& strides,
    const padding& pad_type,
    const im2col_filter_matrix& filter_mat,
    const tensor& input)
{
    assertion(filter_mat.filter_shape_.depth_ == input.shape().depth_,
        "invalid filter depth");

    const auto conv_cfg = preprocess_transpose_convolution(
        filter_mat.filter_shape_.without_depth(),
        strides, pad_type, input.shape().height_, input.shape().width_);

    const std::size_t out_height = conv_cfg.out_height_;
    const std::size_t out_width = conv_cfg.out_width_;

    const auto in_padded = pad_tensor(0,
        conv_cfg.pad_top_, conv_cfg.pad_bottom_, conv_cfg.pad_left_, conv_cfg.pad_right_,
        input);

    return convolve_im2col_transpose(
        out_height, out_width,
        strides.height_, strides.width_,
        filter_mat, in_padded);
}

inline tensor transpose_convolve3D(
    const shape3& strides,
    const padding& pad_type,
    const im2col_filter_matrix& filter_mat,
    const tensor& input)
{
    assertion(filter_mat.filter_shape_.depth_ == input.shape().depth_,
        "invalid filter depth");

    const auto conv_cfg = preprocess_transpose_convolution3D(
        filter_mat.filter_shape_.shape3_without_depth(),
        strides, pad_type, input.dim4(), input.height(), input.width()); //channels last only!!!

    const std::size_t out_height = conv_cfg.out_height_;
    const std::size_t out_width = conv_cfg.out_width_;
    const std::size_t out_depth = conv_cfg.out_depth_;

    const auto in_padded = pad_tensor3D(0,
                                        conv_cfg.pad_top_, conv_cfg.pad_bottom_,
                                        conv_cfg.pad_left_, conv_cfg.pad_right_,
                                        conv_cfg.pad_front_, conv_cfg.pad_back_,
                                        input);

    return convolve_vol2col_transpose(
        out_height, out_width, out_depth,
        strides.height_, strides.width_, strides.depth_,
        filter_mat, in_padded);
}


} } // namespace fdeep, namespace internal
