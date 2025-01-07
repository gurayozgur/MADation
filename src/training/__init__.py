from .trainer import TrainerClip
from .losses import ElasticArcFace, ElasticCosFace, ArcFace, CosFace, AdaFace, BinaryCrossEntropyHeader

def get_trainer(rank, world_size, model_name, model, trainset, dataloader, train_sampler, training_type, config, header, test_dataloader=None, test_sampler=None):
    if model_name == "clip":
        trainer = TrainerClip(rank, world_size, model, trainset, dataloader, train_sampler, training_type, config, header, test_dataloader, test_sampler)
    else:
        raise ValueError()

    return trainer


def get_header(rank, backbone_out_dim, plus=True, **kwargs):
    loss = kwargs["loss"]

    if loss == "ElasticArcFace":
        header = ElasticArcFace(in_features=backbone_out_dim, out_features=kwargs["num_classes"],
                                s=kwargs["s"], m=kwargs["m"], std=kwargs["std"]).to(rank)
    elif loss == "ElasticArcFacePlus":
        header = ElasticArcFace(in_features=backbone_out_dim, out_features=kwargs["num_classes"],
                                s=kwargs["s"], m=kwargs["m"], std=kwargs["std"], plus=plus).to(rank)
    elif loss == "ElasticCosFace":
        header = ElasticCosFace(in_features=backbone_out_dim, out_features=kwargs["num_classes"],
                                s=kwargs["s"] , m=kwargs["m"], std=kwargs["std"]).to(rank)
    elif loss == "ElasticCosFacePlus":
        header = ElasticCosFace(in_features=backbone_out_dim, out_features=kwargs["num_classes"],
                                s=kwargs["s"] , m=kwargs["m"], std=kwargs["std"], plus=plus).to(rank)
    elif loss == "ArcFace":
        header = ArcFace(in_features=backbone_out_dim, out_features=kwargs["num_classes"],
                         s=kwargs["s"], m=kwargs["m"]).to(rank)
    elif loss == "CosFace":
        header = CosFace(in_features=backbone_out_dim, out_features=kwargs["num_classes"],
                         s=kwargs["s"] , m=kwargs["m"]).to(rank)
    elif loss == "AdaFace":
        header = AdaFace(embedding_size=backbone_out_dim, classnum=kwargs["num_classes"]).to(rank)
    elif loss == "BinaryCrossEntropy":
        header = BinaryCrossEntropyHeader(in_features=backbone_out_dim, out_features=kwargs["num_classes"]).to(rank)
    else:
        raise ValueError()

    return header

