using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Vicodin
{
    class InputData
    {
        [ColumnName("PixelValues")]
        [VectorType(784)]
        public float[] PixelValues;

        [LoadColumn(0)]
        public float Number;

        public InputData(float[] input)
        {
            PixelValues = input;
        }
    }
}
