using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Vicodin
{
    class OutputData
    {
        [ColumnName("Score")]
        public float[] Score;
    }
}
