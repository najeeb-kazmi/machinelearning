// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.Trainers.Ensemble;

[assembly: LoadableClass(typeof(DisagreementDiversityMeasure), null, typeof(SignatureEnsembleDiversityMeasure),
    DisagreementDiversityMeasure.UserName, DisagreementDiversityMeasure.LoadName)]

namespace Microsoft.ML.Trainers.Ensemble
{
    internal sealed class DisagreementDiversityMeasure : BaseDisagreementDiversityMeasure<float>, IBinaryDiversityMeasure
    {
        public const string UserName = "Disagreement Diversity Measure";
        public const string LoadName = "DisagreementDiversityMeasure";

        protected override float GetDifference(in float valueX, in float valueY)
        {
            return (valueX > 0 && valueY < 0 || valueX < 0 && valueY > 0) ? 1 : 0;
        }
    }
}
