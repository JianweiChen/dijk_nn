import torch
from common import *

K1 = 64

def to_line_group_feature(x):
    K2 = K1 - 1
    seqcount, _ = x.shape
    if seqcount <= K2:
        x = tensor_to_shape_with_zero(x, (K1, 2))
        return x
    n_output = seqcount // K2 + 1
    h1 = seqcount // n_output
    hs = [h1]*n_output
    h2 = sum(hs)
    h3 = seqcount - h2
    i = 0
    while h3>0:
        hs[i] += 1
        h3 -= 1
        i += 1
    hcum = 0
    output_tensor_list = []
    for h in hs:
        ka, kb = hcum-1, hcum+h
        ka = max(0, ka)
        xpart = x[ka: kb, ...]
        xpart = tensor_to_shape_with_zero(xpart, (K1, 2))
        output_tensor_list.append(xpart)
        hcum += h
    output_tensor = torch.cat(output_tensor_list, axis=1)
    return output_tensor

def save_guidepost_line_tensor():
    df = pd.read_csv(data_path_("history/geo_stop_all_country.csv"))
    df = df.assign(location=lambda xdf: xdf.apply(axis=1, func=lambda row: (row.lng, row.lat)))
    df = df.groupby("lineid").location.unique()
    line_tensor = torch.cat(
        df.apply(lambda x: torch.tensor(x.tolist())).apply(to_line_group_feature).tolist(), axis=1)
    line_tensor = line_tensor.reshape((K1, -1, 2)).permute(1, 0, 2)
    torch.save(line_tensor, pathlib.Path(data_path_("line_tensor.pt")).open('wb'))

def load_guodepost_line_tensor():
    line_tensor = torch.load(
        pathlib.Path(data_path_("line_tensor.pt")).open('rb'))
    return line_tensor

def filter_line_tensor(x_tensor, center, km_a):
    _temp = line_tensor.to_sparse(2)
    kdtree = KDTree(_temp.values().numpy())
    x_idxs = torch.tensor(kdtree.query_ball_point(center, km_a/110))
    ls_tensor = _temp.indices().T.index_select(0, x_idxs)[:, 0].unique()
    output_tensor = line_tensor.index_select(0, ls_tensor)
    return output_tensor

STATIC_LOCATION_BUAA                    = [116.34762, 39.97692]
STATIC_LOCATION_SHANGHAI_SUZHOU_MIDDLE  = [120.95164, 31.28841]

_material_tensor_config = [
    (
        'shanghai_suzhou_middle_50km',
        STATIC_LOCATION_SHANGHAI_SUZHOU_MIDDLE,
        50,
    ),
    (
        "buaa_50km",
        STATIC_LOCATION_BUAA,
        50,
    )
]
material_tensor_config = {
    _[0]: _ for _ in _material_tensor_config
}

def save_material_tensor(name, line_tensor):

    line_tensor_sparse_view = line_tensor.to_sparse(2)
    print("line_tensor_sparse_view")
    x0 = line_tensor
    x1 = line_tensor_sparse_view.values()
    x2 = line_tensor_sparse_view.indices().T
    _temp = torch.cat([x2, x2.roll(-1, 0)], axis=1)
    _temp = 1-(_temp[:, 0] - _temp[:, 2]).type(torch.BoolTensor).type(torch.IntTensor)
    _temp = _temp.to_sparse().indices().squeeze()
    x3 = torch.stack([_temp, _temp+1]).T
    print("x3")
    _temp = pd.Series(KDTree(x1.numpy()).query_ball_point(x1.numpy(), 0.8/110)) \
        .explode().reset_index() \
        .to_numpy() \
        .astype(np.int32)
    _temp = torch.tensor(_temp)
    x4 = _temp.index_select(0, (
        _temp[:, 0]-_temp[:, 1]).type(torch.BoolTensor).type(torch.IntTensor).to_sparse().indices().squeeze())
    print("x4")
    _temp = torch.cat([
        x1.index_select(0, x3[..., 0]),
        x1.index_select(0, x3[..., 1])
    ], axis=1)
    x5 = torch_mht(_temp)
    _temp = torch.cat([
        x1.index_select(0, x4[..., 0]),
        x1.index_select(0, x4[..., 1])
    ], axis=1)
    x6 = torch_mht(_temp)
    x7 = torch.cat([x3, x4], axis=0)
    x8 = torch.stack([
        torch.concat([x5, x6], axis=0),
        torch.concat([x5 / 45, x6/4.5+5/60], axis=0)
    ], axis=1)

    _temp = torch.stack([
        x2[..., 0].index_select(0, x7[..., 0]),
        x2[..., 0].index_select(0, x7[..., 1]),
    ], axis=1)
    _temp2 = (_temp[..., 0] - _temp[..., 1]).type(torch.BoolTensor).type(torch.IntTensor).to_sparse().indices().squeeze()
    _temp = _temp.index_select(0, _temp2)

    _temp = pd.DataFrame(_temp, columns=['src', 'tgt']) \
        .groupby(['src', 'tgt']).size().reset_index()[['src', 'tgt']] \
        .to_numpy()
    x9 = torch.tensor(_temp)
    dirpath = pathlib.Path(data_path_(f"material_tensor_{name}"))
    dirpath.mkdir(parents=True, exist_ok=True)
    output_tensor_ids = [0, 1, 2, 7, 8, 9]
    for _id in output_tensor_ids:
        xx = locals()[f"x{_id}"]
        torch.save(xx, (dirpath/f'x{_id}.pt').open('wb'))

if __name__ == '__main__':
    # save_guidepost_line_tensor()
    line_tensor = load_guodepost_line_tensor()
    name = 'shanghai_suzhou_middle_50km'
    name = 'buaa_50km'
    name, center, km_a = material_tensor_config[name]
    line_tensor = filter_line_tensor(line_tensor, center, km_a)

    save_material_tensor(name, line_tensor)
