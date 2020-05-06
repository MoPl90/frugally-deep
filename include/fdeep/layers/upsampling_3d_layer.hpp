// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class upsampling_3d_layer : public layer
{
public:
    explicit upsampling_3d_layer(const std::string& name,
        const shape3& scale_factor, const std::string& interpolation) :
    layer(name),
    scale_factor_(scale_factor),
    interpolation_(interpolation)
    {
        assertion(interpolation == "nearest" || interpolation == "bilinear",
            "Invalid interpolation method: " + interpolation
        );
    }
protected:
    tensors apply_impl(const tensors& inputs) const override final
    {
        const auto& input = single_tensor_from_tensors(inputs);
        if (interpolation_ == "nearest")
        {
            return {upsampling3d_nearest(input)};
        }
        else if (interpolation_ == "bilinear")
        {
            return {upsampling3d_bilinear(input)};
        }
        else
        {
            raise_error("Invalid interpolation method: " + interpolation_);
            return inputs;
        }
    }
    shape3 scale_factor_;
    std::string interpolation_;
    tensor upsampling3d_nearest(const tensor& in_vol) const
    {
        tensor out_vol(tensor_shape(
            in_vol.dim4() * scale_factor_.height_,
            in_vol.height() * scale_factor_.width_,
            in_vol.width() * scale_factor_.depth_,
            in_vol.depth()), 0);
        for (std::size_t dim4 = 0; dim4 < out_vol.dim4(); ++dim4)
        {
            std::size_t dim4_in = dim4 / scale_factor_.height_;
            for (std::size_t y = 0; y < out_vol.shape().height_; ++y)
            {
                std::size_t y_in = y / scale_factor_.width_;
                for (std::size_t x = 0; x < out_vol.width(); ++x)
                {
                    std::size_t x_in = x / scale_factor_.width_;
                    for (std::size_t z = 0; z < in_vol.depth(); ++z)
                    {
                        out_vol.set(tensor_pos(dim4, y, x, z), in_vol.get(tensor_pos(dim4_in, y_in, x_in, z)));
                    }
                }
            }
        }
        return out_vol;
    }
    float_type get_interpolated_bilinearly(const tensor& t,
    float_type dim4, float_type y, float_type x, std::size_t z) const
    {
        dim4 = fplus::max(0, dim4);
        y = fplus::max(0, y);
        x = fplus::max(0, x);
        y = fplus::min(y, t.height());
        x = fplus::min(x, t.width());
        std::size_t dim4_front = static_cast<std::size_t>(fplus::max(0, fplus::floor(dim4)));
        std::size_t dim4_back = static_cast<std::size_t>(fplus::min(t.dim4() - 1, dim4_front + 1));
        std::size_t y_top = static_cast<std::size_t>(fplus::max(0, fplus::floor(y)));
        std::size_t y_bottom = static_cast<std::size_t>(fplus::min(t.height() - 1, y_top + 1));
        std::size_t x_left = static_cast<std::size_t>(fplus::max(0, fplus::floor(x)));
        std::size_t x_right = static_cast<std::size_t>(fplus::min(t.width() - 1, x_left + 1));
        const auto val_front_top_left = t.get(tensor_pos(dim4_front, y_top, x_left, z));
        const auto val_front_top_right = t.get(tensor_pos(dim4_front, y_top, x_right, z));
        const auto val_back_top_left = t.get(tensor_pos(dim4_back, y_top, x_left, z));
        const auto val_back_top_right = t.get(tensor_pos(dim4_back, y_top, x_right, z));
        const auto val_front_bottom_left = t.get(tensor_pos(dim4_front, y_bottom, x_left, z));
        const auto val_front_bottom_right = t.get(tensor_pos(dim4_front, y_bottom, x_right, z));
        const auto val_back_bottom_left = t.get(tensor_pos(dim4_back, y_bottom, x_left, z));
        const auto val_back_bottom_right = t.get(tensor_pos(dim4_back, y_bottom, x_right, z));
        const auto dim4_factor_front = static_cast<float_type>(dim4_back) - dim4;
        const auto dim4_factor_back = 1.0 - dim4_factor_front;
        const auto y_factor_top = static_cast<float_type>(y_bottom) - y;
        const auto y_factor_bottom = 1.0 - y_factor_top;
        const auto x_factor_left = static_cast<float_type>(x_right) - x;
        const auto x_factor_right = 1.0 - x_factor_left;
        return static_cast<float_type>(
            dim4_factor_front * y_factor_top * x_factor_left * val_front_top_left +
            dim4_factor_front * y_factor_top * x_factor_right * val_front_top_right +
            dim4_factor_back * y_factor_top * x_factor_left * val_back_top_left +
            dim4_factor_back * y_factor_top * x_factor_right * val_back_top_right +
            dim4_factor_front *  y_factor_bottom * x_factor_left * val_front_bottom_left +
            dim4_factor_front * y_factor_bottom * x_factor_right * val_front_bottom_right +
            dim4_factor_back *  y_factor_bottom * x_factor_left * val_back_bottom_left +
            dim4_factor_back * y_factor_bottom * x_factor_right * val_back_bottom_right);
    }
    tensor upsampling3d_bilinear(const tensor& in_vol) const
    {
        tensor out_vol(tensor_shape(
            in_vol.dim4() * scale_factor_.height_,
            in_vol.height() * scale_factor_.width_,
            in_vol.width() * scale_factor_.depth_,
            in_vol.depth()), 0);
        
        for (std::size_t dim4 = 0; dim4 < out_vol.dim4(); ++dim4)
        {
            const auto dim4_in = (static_cast<float_type>(dim4) + 0.5f) / static_cast<float_type>(scale_factor_.height_) - 0.5f;
            for (std::size_t y = 0; y < out_vol.height(); ++y)
            {
                const auto y_in = (static_cast<float_type>(y) + 0.5f) / static_cast<float_type>(scale_factor_.width_) - 0.5f;
                for (std::size_t x = 0; x < out_vol.width(); ++x)
                {
                    const auto x_in = (static_cast<float_type>(x) + 0.5f) / static_cast<float_type>(scale_factor_.depth_) - 0.5f;
                    for (std::size_t z = 0; z < in_vol.depth(); ++z)
                    {
                        out_vol.set(tensor_pos(dim4, y, x, z),
                            get_interpolated_bilinearly(in_vol, dim4_in, y_in, x_in, z));
                    }
                }
            }
        }
        return out_vol;
    }
};

} } // namespace fdeep, namespace internal
