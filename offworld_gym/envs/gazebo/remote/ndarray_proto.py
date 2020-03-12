from io import BytesIO
import numpy as np
import cloudpickle

from offworld_gym.envs.gazebo.remote.protobuf.remote_env_pb2 import NDArray

# logic for marshalling numpy ndarrays as protobuf messages


def ndarray_to_proto(nda):
    # nda_bytes = BytesIO()
    # np.save(nda_bytes, nda, allow_pickle=True)
    # return NDArray(ndarray=nda_bytes.getvalue())
    return cloudpickle.dumps(nda)


def proto_to_ndarray(nda_proto):
    # nda_bytes = BytesIO(nda_proto.ndarray)
    # return np.load(nda_bytes, allow_pickle=True)
    return cloudpickle.loads(nda_proto)