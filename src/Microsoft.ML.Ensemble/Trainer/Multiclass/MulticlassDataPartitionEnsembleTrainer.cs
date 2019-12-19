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

[assembly: LoadableClass(MulticlassDataPartitionEnsembleTrainer.Summary, typeof(MulticlassDataPartitionEnsembleTrainer),
    typeof(MulticlassDataPartitionEnsembleTrainer.Arguments),
    new[] { typeof(SignatureMulticlassClassifierTrainer), typeof(SignatureTrainer) },
    MulticlassDataPartitionEnsembleTrainer.UserNameValue,
    MulticlassDataPartitionEnsembleTrainer.LoadNameValue)]

[assembly: LoadableClass(typeof(MulticlassDataPartitionEnsembleTrainer), typeof(MulticlassDataPartitionEnsembleTrainer.Arguments),
    typeof(SignatureModelCombiner), "Multiclass Classification Ensemble Model Combiner", MulticlassDataPartitionEnsembleTrainer.LoadNameValue)]

namespace Microsoft.ML.Trainers.Ensemble
{
    using TVectorPredictor = IPredictorProducing<VBuffer<float>>;
    using TVectorTrainer = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<VBuffer<float>>>, IPredictorProducing<VBuffer<float>>>;

    /// <summary>
    /// A generic ensemble classifier for multi-class classification
    /// </summary>
    internal sealed class MulticlassDataPartitionEnsembleTrainer :
        EnsembleTrainerBase<VBuffer<float>,
        IMulticlassSubModelSelector, IMulticlassOutputCombiner, MulticlassPredictionTransformer<EnsembleMulticlassModelParameters>, EnsembleMulticlassModelParameters>,
        IModelCombiner
    {
        public const string LoadNameValue = "WeightedEnsembleMulticlass";
        public const string UserNameValue = "Multi-class Parallel Ensemble (bagging, stacking, etc)";
        public const string Summary = "A generic ensemble classifier for multi-class classification.";

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "Algorithm to prune the base learners for selective Ensemble", ShortName = "pt", SortOrder = 4)]
            [TGUI(Label = "Sub-Model Selector(pruning) Type", Description = "Algorithm to prune the base learners for selective Ensemble")]
            public ISupportMulticlassSubModelSelectorFactory SubModelSelectorType = new AllSelectorMulticlassFactory();

            [Argument(ArgumentType.Multiple, HelpText = "Output combiner", ShortName = "oc", SortOrder = 5)]
            [TGUI(Label = "Output combiner", Description = "Output combiner type")]
            public ISupportMulticlassOutputCombinerFactory OutputCombiner = new MultiMedian.Options();

            // REVIEW: If we make this public again it should be an *estimator* of this type of predictor, rather than the (deprecated) ITrainer.
            [Argument(ArgumentType.Multiple, HelpText = "Base predictor type", ShortName = "bp,basePredictorTypes", SortOrder = 1, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, SignatureType = typeof(SignatureMulticlassClassifierTrainer))]
            internal IComponentFactory<TVectorTrainer>[] BasePredictors;

            internal override IComponentFactory<TVectorTrainer>[] GetPredictorFactories() => BasePredictors;

            public Arguments()
            {
                BasePredictors = new[]
                {
                    // Note that this illustrates a fundamnetal problem with the mixture of `ITrainer` and `ITrainerEstimator`
                    // present in this class. The options to the estimator have no way of being communicated to the `ITrainer`
                    // implementation, so there is a fundamnetal disconnect if someone chooses to ever use the *estimator* with
                    // non-default column names. Unfortuantely no method of resolving this temporary strikes me as being any
                    // less laborious than the proper fix, which is that this "meta" component should itself be a trainer
                    // estimator, as opposed to a regular trainer.
                    ComponentFactoryUtils.CreateFromFunction(env => new LbfgsMaximumEntropyMulticlassTrainer(env, LabelColumnName, FeatureColumnName))
                };
            }
        }

        private readonly ISupportMulticlassOutputCombinerFactory _outputCombiner;

        public MulticlassDataPartitionEnsembleTrainer(IHostEnvironment env, Arguments args)
            : base(args, env, LoadNameValue, TrainerUtils.MakeU4ScalarColumn(args.LabelColumnName))
        {
            SubModelSelector = args.SubModelSelectorType.CreateComponent(Host);
            _outputCombiner = args.OutputCombiner;
            Combiner = args.OutputCombiner.CreateComponent(Host);
        }

        private MulticlassDataPartitionEnsembleTrainer(IHostEnvironment env, Arguments args, PredictionKind predictionKind)
            : this(env, args)
        {
            Host.CheckParam(predictionKind == PredictionKind.MulticlassClassification, nameof(PredictionKind));
        }

        private protected override PredictionKind PredictionKind => PredictionKind.MulticlassClassification;

        private protected override EnsembleMulticlassModelParameters CreatePredictor()
        {
            return new EnsembleMulticlassModelParameters(Host, CreateModels<TVectorPredictor>(Models), Combiner as IMulticlassOutputCombiner);
        }

        public IPredictor CombineModels(IEnumerable<IPredictor> models)
        {
            Host.CheckValue(models, nameof(models));
            Host.CheckParam(models.All(m => m is TVectorPredictor), nameof(models));

            var combiner = _outputCombiner.CreateComponent(Host);
            var predictor = new EnsembleMulticlassModelParameters(Host,
                models.Select(k => new FeatureSubsetModel<VBuffer<float>>((TVectorPredictor)k)).ToArray(),
                combiner);
            return predictor;
        }

        private protected override void CheckLabel(RoleMappedData data)
        {
            Contracts.AssertValue(data);
            data.CheckMulticlassLabel(out int numClasses);
        }

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            bool success = inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol);
            Contracts.Assert(success);

            var metadata = new SchemaShape(labelCol.Annotations.Where(x => x.Name == AnnotationUtils.Kinds.KeyValues)
                .Concat(AnnotationUtils.GetTrainerOutputAnnotation()));
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.AnnotationsForMulticlassScoreColumn(labelCol))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.UInt32, true, metadata)
            };
        }

        private protected override MulticlassPredictionTransformer<EnsembleMulticlassModelParameters>
            MakeTransformer(EnsembleMulticlassModelParameters model, DataViewSchema trainSchema)
            => new MulticlassPredictionTransformer<EnsembleMulticlassModelParameters>(Host, model, trainSchema, FeatureColumn.Name, LabelColumn.Name);
    }
}
