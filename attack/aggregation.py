import numpy as np
import json
import os
import importlib

def aggregation(args, labels, name=None):
    # Output files used for the vote.
    if name == None:
        filenames = ["shard-{}-{}.npy".format(i, args.requests) for i in range(args.shards)]
    else:
        filenames = ["shard-{}-{}_{}.npy".format(i, args.requests, name) for i in range(args.shards)]

    # Concatenate output files.
    outputs = []
    for filename in filenames:
        outputs.append(
            np.load(
                os.path.join(args.path + "SNO_{}/outputs".format(args.shards), filename),
                allow_pickle=True,
            )
        )
    outputs = np.array(outputs)

    # Compute weight vector based on given strategy.
    weights = (
        1 / outputs.shape[0] * np.ones((outputs.shape[0],))
    )  # pylint: disable=unsubscriptable-object

    # Tensor contraction of outputs and weights (on the shard dimension).
    votes = np.argmax(
        np.tensordot(weights.reshape(1, weights.shape[0]), outputs, axes=1), axis=2
    ).reshape(
        (outputs.shape[1],)
    )  # pylint: disable=unsubscriptable-object

    # Compute and print accuracy.
    accuracy = (
        np.where(votes == labels)[0].shape[0] / outputs.shape[1]
    )  # pylint: disable=unsubscriptable-object
    print(accuracy)
