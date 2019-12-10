using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.EntryPoints
{
    using TScalarTrainer = ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<float>>, IPredictorProducing<float>>;

    [TlcModule.ComponentKind("EnsembleBinaryBasePredictor")]
    [BestFriend]
    internal interface IEnsembleBinaryBasePredictorFactory : IComponentFactory<TScalarTrainer>
    {

    }

}
