// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Ensemble;

[assembly: LoadableClass(EnsembleTrainer.Summary, typeof(EnsembleTrainer), typeof(EnsembleTrainer.Arguments),
    new[] { typeof(SignatureBinaryClassifierTrainer), typeof(SignatureTrainer) },
    EnsembleTrainer.UserNameValue, EnsembleTrainer.LoadNameValue, "pe", "ParallelEnsemble")]

[assembly: LoadableClass(typeof(EnsembleTrainer), typeof(EnsembleTrainer.Arguments), typeof(SignatureModelCombiner),
    "Binary Classification Ensemble Model Combiner", EnsembleTrainer.LoadNameValue, "pe", "ParallelEnsemble")]

namespace Microsoft.ML.Trainers.Ensemble
{
    using TDistPredictor = IDistPredictorProducing<float, float>;
    using TScalarPredictor = IPredictorProducing<float>;
    using TScalarTrainer = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>>;

    /// <summary>
    /// A generic ensemble trainer for binary classification.
    /// </summary>
    internal sealed class EnsembleTrainer : EnsembleTrainerBase<float,
        IBinarySubModelSelector, IBinaryOutputCombiner, BinaryPredictionTransformer<EnsembleModelParametersBase<float>>, EnsembleModelParametersBase<float>>,
        IModelCombiner
    {
        public const string LoadNameValue = "WeightedEnsemble";
        public const string UserNameValue = "Parallel Ensemble (bagging, stacking, etc)";
        public const string Summary = "A generic ensemble classifier for binary classification.";

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Algorithm to prune the base learners for selective Ensemble", ShortName = "pt", SortOrder = 4)]
            [TGUI(Label = "Sub-Model Selector(pruning) Type",
                Description = "Algorithm to prune the base learners for selective Ensemble")]
            public ISupportBinarySubModelSelectorFactory SubModelSelectorType = new AllSelectorFactory();

            [Argument(ArgumentType.Multiple, HelpText = "Output combiner", ShortName = "oc", SortOrder = 5)]
            [TGUI(Label = "Output combiner", Description = "Output combiner type")]
            public ISupportBinaryOutputCombinerFactory OutputCombiner = new MedianFactory();

            // REVIEW: If we make this public again it should be an *estimator* of this type of predictor, rather than the (deprecated) ITrainer.
            [Argument(ArgumentType.Multiple, HelpText = "Base predictor type", ShortName = "bp,basePredictorTypes", SortOrder = 1, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, SignatureType = typeof(SignatureBinaryClassifierTrainer))]
            public IComponentFactory<TScalarTrainer>[] BasePredictors;

            internal override IComponentFactory<TScalarTrainer>[] GetPredictorFactories() => BasePredictors;

            public Arguments()
            {
                BasePredictors = new[]
                {
                    ComponentFactoryUtils.CreateFromFunction(env => new LinearSvmTrainer(env, LabelColumnName, FeatureColumnName))
                };
            }
        }

        private readonly ISupportBinaryOutputCombinerFactory _outputCombiner;

        public EnsembleTrainer(IHostEnvironment env, Arguments args)
            : base(args, env, LoadNameValue, TrainerUtils.MakeBoolScalarLabel(args.LabelColumnName))
        {
            SubModelSelector = args.SubModelSelectorType.CreateComponent(Host);
            _outputCombiner = args.OutputCombiner;
            Combiner = args.OutputCombiner.CreateComponent(Host);
        }

        private EnsembleTrainer(IHostEnvironment env, Arguments args, PredictionKind predictionKind)
            : this(env, args)
        {
            Host.CheckParam(predictionKind == PredictionKind.BinaryClassification, nameof(PredictionKind));
        }

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private protected override EnsembleModelParametersBase<float> CreatePredictor()
        {
            if (Models.All(m => m.Predictor is TDistPredictor))
                return new EnsembleDistributionModelParameters(Host, PredictionKind, CreateModels<TDistPredictor>(Models), Combiner);
            return new EnsembleModelParameters(Host, PredictionKind, CreateModels<TScalarPredictor>(Models), Combiner);
        }

        public IPredictor CombineModels(IEnumerable<IPredictor> models)
        {
            Host.CheckValue(models, nameof(models));

            var combiner = _outputCombiner.CreateComponent(Host);
            var p = models.First();
            if (p is TDistPredictor)
            {
                Host.CheckParam(models.All(m => m is TDistPredictor), nameof(models));
                return new EnsembleDistributionModelParameters(Host, p.PredictionKind,
                    models.Select(k => new FeatureSubsetModel<float>((TDistPredictor)k)).ToArray(), combiner);
            }

            Host.CheckParam(models.All(m => m is TScalarPredictor), nameof(models));
            return new EnsembleModelParameters(Host, p.PredictionKind,
                    models.Select(k => new FeatureSubsetModel<float>((TScalarPredictor)k)).ToArray(), combiner);
        }

        private protected override void CheckLabel(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            data.CheckBinaryLabel();
        }

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }

        private protected override BinaryPredictionTransformer<EnsembleModelParametersBase<float>>
            MakeTransformer(EnsembleModelParametersBase<float> model, DataViewSchema trainSchema)
            => new BinaryPredictionTransformer<EnsembleModelParametersBase<float>>(Host, model, trainSchema, FeatureColumn.Name);
}