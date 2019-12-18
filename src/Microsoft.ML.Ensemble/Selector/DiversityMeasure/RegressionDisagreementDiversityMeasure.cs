﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.Trainers.Ensemble;

[assembly: LoadableClass(typeof(RegressionDisagreementDiversityMeasure), null, typeof(SignatureEnsembleDiversityMeasure),
    DisagreementDiversityMeasure.UserName, RegressionDisagreementDiversityMeasure.LoadName)]

namespace Microsoft.ML.Trainers.Ensemble
{
    internal sealed class RegressionDisagreementDiversityMeasure : BaseDisagreementDiversityMeasure<float>, IRegressionDiversityMeasure
    {
        public const string LoadName = "RegressionDisagreementDiversityMeasure";

        protected override float GetDifference(in float valueX, in float valueY)
        {
            return Math.Abs(valueX - valueY);
        }
    }
}
