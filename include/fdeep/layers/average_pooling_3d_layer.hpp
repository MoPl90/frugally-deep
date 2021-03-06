// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/pooling_3d_layer.hpp"

#include <limits>
#include <string>

namespace fdeep { namespace internal
{

FDEEP_FORCE_INLINE tensor average_pool_3d(
    std::size_t pool_height, std::size_t pool_width, std::size_t pool_depth,
    std::size_t strides_y, std::size_t strides_x, std::size_t strides_z,
    bool channels_first,
    padding pad_type,
    const tensor& in)
{
    const float_type invalid = std::numeric_limits<float_type>::lowest();

    const std::size_t feature_count = channels_first
        ? in.shape().size_dim_4_
        : in.shape().depth_
        ;

    const std::size_t in_height = channels_first
        ? in.shape().height_
        : in.shape().size_dim_4_
        ;

    const std::size_t in_width = channels_first
        ? in.shape().width_
        : in.shape().height_
        ;
    
    const std::size_t in_depth = channels_first
        ? in.shape().depth_
        : in.shape().width_
        ;
    
    const auto conv_cfg = preprocess_convolution3D(
        shape3(pool_height, pool_width, pool_depth),
        shape3(strides_y, strides_x, strides_z),
        pad_type, in_height, in_width, in_depth);

    int pad_top_int = static_cast<int>(conv_cfg.pad_top_);
    int pad_left_int = static_cast<int>(conv_cfg.pad_left_);
    int pad_front_int = static_cast<int>(conv_cfg.pad_front_);
    const std::size_t out_height = conv_cfg.out_height_;
    const std::size_t out_width = conv_cfg.out_width_;
    const std::size_t out_depth = conv_cfg.out_depth_;

    // todo: Do we still need to support this, check test_model_exhaustive
    if (channels_first)
    {
        tensor out(tensor_shape(feature_count, out_height, out_width, out_depth), 0);

        for (std::size_t f = 0; f < feature_count; ++f)
        {
            for (std::size_t y = 0; y < out_height; ++y)
            {
                for (std::size_t x = 0; x < out_width; ++x)
                {
                    for (std::size_t z = 0; z < out_depth; ++z)
                    {
                        float_type val = 0;
                        std::size_t divisor = 0;
                        for (std::size_t yf = 0; yf < pool_height; ++yf)
                        {
                            int in_get_y = static_cast<int>(strides_y * y + yf) - pad_top_int;
                            for (std::size_t xf = 0; xf < pool_width; ++xf)
                            {
                                int in_get_x = static_cast<int>(strides_x * x + xf) - pad_left_int;
                                for (std::size_t zf = 0; zf < pool_depth; ++zf)
                                {
                                    int in_get_z = static_cast<int>(strides_z * z + zf) - pad_front_int;
                                    const auto current = in.get_y_x_z_padded(invalid, f, in_get_y, in_get_x, in_get_z);
                                    if (current != invalid)
                                    {
                                        val += current;
                                        divisor += 1;
                                    }
                                }
                            }
                        }
                        out.set(tensor_pos(f, y, x, z), val / static_cast<float_type>(divisor));
                    }
                }
            }
        }
        return out;
    }
    else
    {
        tensor out(
            tensor_shape_with_changed_rank(
                tensor_shape(out_height, out_width, out_depth, feature_count),
                in.shape().rank()),
            0);

        for (std::size_t y = 0; y < out_height; ++y)
        {
            for (std::size_t x = 0; x < out_width; ++x)
            {
                for (std::size_t z = 0; z < out_depth; ++z)
                {
                    for (std::size_t f = 0; f <feature_count; ++f)
                    {
                        float_type val = 0.;
                        std::size_t divisor = 0;
                        for (std::size_t yf = 0; yf < pool_height; ++yf)
                        {
                            int in_get_y = static_cast<int>(strides_y * y + yf) - pad_top_int;
                            for (std::size_t xf = 0; xf < pool_width; ++xf)
                            {
                                int in_get_x = static_cast<int>(strides_x * x + xf) - pad_left_int;
                                for (std::size_t zf = 0; zf < pool_depth; ++zf)
                                {
                                    int in_get_z = static_cast<int>(strides_z * z + zf) - pad_front_int;
                                    const auto current = in.get_dim4_y_x_padded(invalid, in_get_y, in_get_x, in_get_z, f);
                                    if (current != invalid)
                                    {
                                        val += current;
                                        divisor += 1;
                                    }
                                }
                            }
                        }
                        out.set_ignore_rank(tensor_pos(y, x, z, f), val / static_cast<float_type>(divisor));
                    }
                }
            }
        }
        return out;
    }
}

class average_pooling_3d_layer : public pooling_3d_layer
{
public:
    explicit average_pooling_3d_layer(const std::string& name,
        const shape3& pool_size, const shape3& strides, bool channels_first,
        padding p) :
        pooling_3d_layer(name, pool_size, strides, channels_first, p)
    {
    }
protected:
    tensor pool(const tensor& in) const override
    
    {
        if (pool_size_ == shape3(2, 2, 2) && strides_ == shape3(2, 2, 2))
            return average_pool_3d(2, 2, 2, 2, 2, 2, channels_first_, padding_, in);
        else if (pool_size_ == shape3(4, 4, 4) && strides_ == shape3(4, 4, 4))
            return average_pool_3d(4, 4, 4, 4, 4, 4, channels_first_, padding_, in);
        else
            return average_pool_3d(
                pool_size_.height_, pool_size_.width_, pool_size_.depth_,
                strides_.height_, strides_.width_, strides_.depth_,
                channels_first_, padding_, in);
    }
};

} } // namespace fdeep, namespace internal
