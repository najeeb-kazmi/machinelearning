// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Numeric;
using Microsoft.ML.Trainers.Ensemble;

[assembly: LoadableClass(typeof(MultiDisagreementDiversityMeasure), null, typeof(SignatureEnsembleDiversityMeasure),
    DisagreementDiversityMeasure.UserName, MultiDisagreementDiversityMeasure.LoadName)]

namespace Microsoft.ML.Trainers.Ensemble
{
    internal sealed class MultiDisagreementDiversityMeasure : BaseDisagreementDiversityMeasure<VBuffer<float>>, IMulticlassDiversityMeasure
    {
        public const string LoadName = "MultiDisagreementDiversityMeasure";

        protected override float GetDifference(in VBuffer<float> valueX, in VBuffer<float> valueY)
        {
            return (VectorUtils.ArgMax(in valueX) != VectorUtils.ArgMax(in valueY)) ? 1 : 0;
        }
    }
}
