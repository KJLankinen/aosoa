#include "variable.h"
#include "aosoa.h"

using namespace aosoa;

// clang-format off
template <size_t Alignment, typename MemOps>
using Balls = StructureOfArrays<
            Alignment,
            MemOps,
            Variable<double, "position_x">,
            Variable<double, "position_y">,
            Variable<double, "position_z">,
            Variable<double, "radius">,
            Variable<float, "color_r">,
            Variable<float, "color_g">,
            Variable<float, "color_b">,
            Variable<uint32_t, "index">,
            Variable<int32_t, "index_distance">,
            Variable<bool, "is_visible">>;
// clang-format on

template <size_t Alignment> using CBalls = Balls<Alignment, CMemoryOperations>;
template <size_t Alignment> using Ball = CBalls<Alignment>::FullRow;
template <size_t Alignment>
using BallAccessor = CBalls<Alignment>::ThisAccessor;
